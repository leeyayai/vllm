# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.multimodal import MultiModalRegistry
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.
    管理 vLLM V1 多模态模型的 encoder 输出缓存。

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    EncoderCacheManager 负责管理多模态 encoder 输出（例如图像的视觉 embedding）
    (such as vision embeddings from images) during request processing. It
    在请求处理期间的生命周期，并提供
    provides memory-aware caching to avoid recomputing encoder outputs when the
    感知内存用量的缓存机制，避免在
    same multimodal inputs appear in different stages of request processing.
    请求处理的不同阶段遇到相同多模态输入时重复计算 encoder 输出。

    This manager is particularly important for:
    此管理器尤其适用于：
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
    - 需要缓存图像 encoder 输出的视觉语言模型（例如 LLaVA）；
      cached
    - Any multimodal model where encoder computation is expensive and
    - encoder 计算开销较大的其他多模态模型，
      cacheable
      且其结果可以缓存。

    The cache operates at the granularity of individual multimodal input items
    缓存以单个多模态输入项为粒度，
    within requests, allowing for fine-grained memory management and enabling
    支持精细的内存管理，也支持
    chunked processing of multimodal inputs.
    对多模态输入进行分块处理。

    Cache is enabled to share embeddings of same multimodal data
    缓存可在不同请求间共享相同多模态数据项（按哈希值识别）的 embedding。
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    当没有空闲空间容纳新 embedding 时，会在分配阶段执行淘汰。
    space for new embeddings.
    Oldest cached embeddings with no request referenced will be first evicted.
    最早缓存且不再被任何请求引用的 embedding 会优先淘汰。

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    注意：EncoderCacheManager 以多模态 embedding 为单位工作，
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    而不是以 encoder token 为单位（即输入序列中表示多模态数据的所有 token）。
    in the input sequence). This means all break/text tokens in-between multimodal
    因此，多模态 embedding 之间的分隔符和文本 token
    embeddings are not considered with respect to the cache size and the number
    不计入 cache 大小和
    of free slots.
    空闲槽位数量。

    Args:
        cache_size: Limit the size of the cache, measured by the number of
        cache_size：缓存容量上限，以输入序列中的 encoder embedding 数量
                    encoder embeddings from the input sequence.
                    衡量。

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        cache_size：encoder embedding 的缓存总容量。
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_free_slots：当前可用的 encoder embedding 缓存容量。
        num_freeable_slots: Capacity that can be immediately reclaimed by
        num_freeable_slots：通过淘汰引用计数为零的条目可以立即回收的容量，
            evicting entries with zero references (in encoder embeddings).
            单位为 encoder embedding。
        cached: Mapping from mm_hash to a set of request IDs that currently
        cached：mm_hash 到当前引用该缓存条目的请求 ID 集合的映射。
            reference the cached entry. If the set is empty, the entry exists
            集合为空时表示条目仍存在，但当前没有请求引用，
            but is not referenced by any request and is eligible for
            因而可以回收。
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
        freeable：由 `(mm_hash, num_encoder_embeds)` 组成的条目列表，表示当前
            whose no current running request is needed and that can be freed to
            没有运行中请求需要、必要时可释放以腾出空间的缓存条目。
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
        freed：自上次调用 get_freed_mm_hashes() 后实际淘汰的 mm_hash 列表。
            last call to get_freed_mm_hashes(). This list is cleared on return.
            返回该列表时会将其清空。
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        # 多模态数据的 mm_hash -> 引用该数据的请求 ID 集合。
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        # 多模态数据的 mm_hash -> 该数据包含的 encoder embedding 数量。
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.
        检查指定多模态输入的 encoder 输出是否已缓存。

        If the encoder output is cached, update `cached` to add the request id
        若 encoder 输出已缓存，则更新 `cached`，将请求 ID 加入
        to the set of request ids that reference the cached encoder output.
        引用该输出的请求集合。
        If the encoder output was previously not referenced by any request,
        如果该输出此前没有被任何请求引用，
        update `freeable` and `num_freeable_slots` accordingly.
        则相应更新 `freeable` 和 `num_freeable_slots`。

        Args:
            request: The request containing the multimodal input
            request：包含该多模态输入的请求。
            input_id: Index of the multimodal input within the request
            input_id：该输入在请求中的索引。

        Returns:
            True if the encoder output for this input is already cached
            若该输入的 encoder 输出已缓存，则返回 True。
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        # 尚未缓存。
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        # 已缓存，但当前没有任何请求引用。
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        检查是否有足够的缓存空间容纳多模态输入。
        If there is, return True and update EncoderCacheManager state.
        若空间足够，则返回 True 并更新 EncoderCacheManager 状态。

        If there is not enough free space in `num_free_slots` but there is
        如果 `num_free_slots` 中的空闲空间不足，但 `num_freeable_slots` 中
        enough reclaimable space in `num_freeable_slots`, entries will be
        有足够可回收空间，则会从 `freeable` 中淘汰条目（将其 mm_hash
        evicted from `freeable` (their mm_hash appended to `freed`) until
        追加到 `freed`），直到有足够空间，然后
        enough space is available, and then this method returns True.
        本方法返回 True。
        Older entries are evicted first.
        较旧的条目优先淘汰。

        Returns False only if the requested number of tokens exceeds both
        仅当请求的 token 数超过空闲容量和可回收容量之和时，
        the free and reclaimable capacities combined.
        才会返回 False。

        Args:
            request: The request containing the multimodal input.
            request：包含该多模态输入的请求。
            input_id: Index of the multimodal input within the request.
            input_id：该输入在请求中的索引。
            encoder_compute_budget: Number of encoder embeddings allowed to be
            encoder_compute_budget：调用本方法时允许计算的 encoder embedding 数量。
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
            num_embeds_to_schedule：调用本方法时已安排分配缓存空间的 encoder
                allocated with cache space when this method is invoked.
                embedding 数量。

        Returns:
            True if there's enough capacity to hold the encoder output for this
            若有足够容量保存该输入的 encoder 输出（必要时先回收 `freeable` 条目），
            input (possibly after reclaiming `freeable` entries); otherwise
            则返回 True；否则
            False.
            返回 False。

        Note: This method does not allocate physical memory for the encoder
        注意：此方法不会为 encoder 输出分配物理内存，
        output but only the state of EncoderCacheManager.
        只会更新 EncoderCacheManager 的状态。
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        # 计算预算不足。
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # Enough free slots
        # 空闲槽位充足。
        if num_embeds <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        # 可回收槽位不足。
        if num_embeds > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots
        # 空闲槽位不足，但可回收槽位充足。
        # NOTE: Eviction takes place here, but physical memory is not freed
        # 注意：此处会执行淘汰，但物理内存不会立即释放，
        # until model runner is notified by the scheduler output.
        # 直到 SchedulerOutput 通知 Model Runner 后才会释放。
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
        return True

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.
        为多模态输入的 encoder 输出分配缓存空间。

        This reserves cache space for storing the encoder output of the
        此方法预留缓存空间以存储指定多模态输入的 encoder 输出。
        specified multimodal input. The actual encoder output storage happens in
        实际的 encoder 输出存储由 Model Runner 完成；
        the model runner; this method updates the manager's bookkeeping.
        此方法只更新管理器的记账状态。

        Note:
            This method assumes can_allocate() returned True for the same input.
            注意：此方法假设同一输入的 can_allocate() 已返回 True。
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # 注意：已调度的 encoder 输入始终应有足够的 encoder cache 空间，
        # that are scheduled since eviction takes place at can_allocate().
        # 因为淘汰已在 can_allocate() 中执行。
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.
        获取请求中所有已缓存的多模态输入 ID。

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        返回 `mm_hash` 存在于缓存映射中的输入 ID 集合。
        This includes entries that are currently unreferenced (and thus present
        其中也包括当前没有引用者（因而位于
        in `freeable`); for such entries, freeing for this request will be a
        `freeable` 中）的条目；对这类条目释放当前请求的引用
        no-op.
        不会产生实际操作。
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)
        释放该请求对 encoder 输入（`mm_data`）的引用。

        When the reference set for the corresponding `mm_hash` becomes empty,
        当对应 `mm_hash` 的引用集合变空时，
        the entry is appended to `freeable` and `num_freeable_slots` is
        会将该条目加入 `freeable`，并增加 `num_freeable_slots`，
        increased by the number of encoder embeddings for that input.
        增量为该输入包含的 encoder embedding 数量。

        The entry is NOT physically freed until capacity is needed (e.g., by
        只有在需要腾出容量时（例如调用
        `can_allocate`).
        `can_allocate`）才会真正释放该条目。
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        # mm_hash 不在缓存中，或该 mm_hash 对应的引用集合为空。
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.
        释放 *request* 持有的所有 encoder 输入缓存引用。

        For each cached input ID, `free_encoder_input` is invoked.
        对每个已缓存输入 ID 调用 `free_encoder_input`。
        The data stays in memory until eviction is triggered by a future
        数据会继续保留在内存中，直到之后的分配操作通过
        attempt allocation called by 'can_allocate'.
        调用 `can_allocate` 触发淘汰。

        Typically called when a request is finished, cancelled, or aborted.
        通常在请求完成、取消或中止时调用。
        """
        input_ids = self.get_cached_input_ids(request).copy()
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.
        获取并清空最近释放的 encoder cache 条目列表。

        Returns:
            List of mm_hash strings that were actually evicted since the last
            返回自上次调用以来实际淘汰的 mm_hash 列表，
            call to be used by the scheduler to notify workers about which
            Scheduler 会使用该列表通知 Worker 哪些
            encoder outputs can be removed from their caches. The internal
            encoder 输出可以从各自缓存中移除。内部
            list is cleared after this call.
            列表会在本次调用后清空。
        """
        freed = self.freed
        self.freed = []
        return freed


def compute_encoder_budget(
    model_config: "ModelConfig",
    scheduler_config: "SchedulerConfig",
    mm_registry: MultiModalRegistry,
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    根据模型和 Scheduler 配置计算 encoder cache 预算。
    configurations.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
        - encoder 执行的计算预算，以输入序列中的 token 数量
            from the input sequence.
          衡量。
        - Space budget for encoder cache size, measured in number of tokens
        - encoder cache 的空间预算，同样以输入序列中的 token 数量
            from the input sequence.
          衡量。
    """
    if mm_registry.supports_multimodal_inputs(model_config):
        max_tokens_by_modality = mm_registry.get_max_tokens_per_item_by_modality(
            model_config
        )

        return compute_mm_encoder_budget(
            scheduler_config,
            max_tokens_by_modality,
        )

    return compute_text_encoder_budget(scheduler_config)


def compute_text_encoder_budget(scheduler_config: "SchedulerConfig") -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    为纯文本模型，根据模型和 Scheduler 配置计算 encoder cache 预算。
    configurations for a text-only model.

    Args:
        scheduler_config: Scheduler configuration.
        scheduler_config：Scheduler 配置。

    Returns:
        - Compute budget for encoder execution, in unit of number of tokens
        - encoder 执行的计算预算，以输入序列中的 token 数量
            in the input sequence.
          为单位。
        - Space budget for encoder cache size, in unit of number of tokens
        - encoder cache 的空间预算，以输入序列中的 token 数量
            in the input sequence.
          为单位。
    """
    # Currently text-only encoder-decoder models are not supported
    # 目前尚不支持纯文本 encoder-decoder 模型。
    return 0, 0


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    max_tokens_by_modality: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    为多模态模型，根据模型和 Scheduler 配置计算 encoder cache 预算。
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        scheduler_config：Scheduler 配置。
        max_tokens_by_modality: The maximum number of tokens for each
        max_tokens_by_modality：每种非文本模态的最大 token 数。
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
        - encoder 执行的计算预算，以输入序列中的 token 数量
            from the input sequence.
          衡量。
        - Space budget for encoder cache size, measured in number of tokens
        - encoder cache 的空间预算，同样以输入序列中的 token 数量
            from the input sequence.
          衡量。
    """

    if not max_tokens_by_modality:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(max_tokens_by_modality.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
# 注意（NickLucche）：这是 encoder-decoder 模型的临时实现，目前此类模型仅
# use the manager for scheduling purposes. Encoder-decoder models will eventually
# 将管理器用于调度。后续 encoder-decoder 模型也会使用缓存，届时本类会
# utilize the cache and this class will fold into EncoderCacheManager, as
# 并入 EncoderCacheManager，届时它与多模态模型之间的差异也会进一步缩小。
# differences with MM models shrink.
class EncoderDecoderCacheManager(EncoderCacheManager):
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # Not enough compute budget
        # 计算预算不足。
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # Enough free slots
        # 空闲槽位充足。
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        # As encoder cache is not used for enc-dec models, we can free the entries here
        # encoder-decoder 模型不使用 encoder cache，因此可以在此释放条目。
        # The actual free happens in the runner, *before* the model is executed.
        # 实际释放由 Runner 在模型执行前完成。
        # Therefore, `freeable` acts as a buffer to free the entries only after the
        # 因此，`freeable` 充当缓冲区，确保条目只会在模型执行后释放，
        # model is executed, mimicking the state transition of `EncoderCacheManager`.
        # 以模拟 `EncoderCacheManager` 的状态转换。
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds
