# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import lcm

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    协调不同 KV cache group 的缓存管理。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # Needs special handling for find_longest_cache_hit if eagle is enabled
        # 启用 Eagle 时，find_longest_cache_hit 需要特殊处理。
        self.use_eagle = use_eagle
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.
        获取该请求需要分配的 block 数量。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            num_tokens: The total number of tokens that need a slot (including
            num_tokens：需要 slot 的 token 总数（包括
                tokens that are already allocated).
                已经分配空间的 token）。
            new_computed_blocks: The new computed blocks just hitting the
            new_computed_blocks：刚刚命中
                prefix caching.
                前缀缓存的已计算 block。
            num_encoder_tokens: The number of encoder tokens for allocating
            num_encoder_tokens：为 cross-attention 分配 block 所需的
                blocks for cross-attention.
                encoder token 数量。
            total_computed_tokens: Include both local and external tokens.
            total_computed_tokens：本地和外部已计算 token 的总数。

        Returns:
            The number of blocks to allocate.
            要分配的 block 数量。
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # 对 cross-attention，根据 encoder 输入 token 数量
                # of blocks based on the number of encoder input tokens.
                # 一次性静态分配所需的 block。
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id, num_encoder_tokens, [], 0
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                )
        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new
        将新计算的 block 添加到请求中；如有外部计算 token，
            blocks for external computed tokens (if any).
        也可以为其分配新 block。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            new_computed_blocks: The new computed blocks just hitting the
            new_computed_blocks：刚刚命中
                prefix cache.
                前缀缓存的已计算 block。
            num_local_computed_tokens: The number of local computed tokens.
            num_local_computed_tokens：本地已计算 token 数。
            num_external_computed_tokens: The number of external computed tokens.
            num_external_computed_tokens：外部已计算 token 数。
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        为请求分配新 block，使其至少拥有 `num_tokens` 个
        token slots.
        token slot。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            num_tokens: The total number of tokens that need a slot (including
            num_tokens：需要 slot 的 token 总数（包括
                tokens that are already allocated).
                已经分配空间的 token）。
            num_encoder_tokens: The number of encoder tokens for allocating
            num_encoder_tokens：为 cross-attention 分配 block 所需的
                blocks for cross-attention.
                encoder token 数量。

        Returns:
            The new allocated blocks.
            新分配的 block。
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.
        缓存该请求的 block。

        Args:
            request: The request.
            request：目标请求。
            num_computed_tokens: The total number of tokens
            num_computed_tokens：需要缓存的 token 总数，
                that need to be cached
                其中包括
                (including tokens that are already cached).
                已经缓存的 token。
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.
        释放该请求的 block。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated
        获取每个 KV cache group 中，所有已分配
        KV cache for each kv cache group.
        KV cache 的请求所共有的前缀 block 数。

        Args:
            running_request_id: The request ID of any running request, used to
            running_request_id：任一运行中请求的 ID，用于
                identify the common prefix blocks.
                识别公共前缀 block。

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
            list[int]：每个 KV cache group 的公共前缀 block 数量。
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace
        从 `blocks` 中移除不再需要的 block，并用
        the removed blocks with null_block.
        null_block 替换被移除的位置。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            total_computed_tokens: The total number of computed tokens, including
            total_computed_tokens：已计算 token 的总数，包括
                local computed tokens and external computed tokens.
                本地和外部计算的 token。
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request.
        获取该请求的 block。
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        pass


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    在前缀缓存被禁用或不受支持时使用的 KV cache coordinator。
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    与 UnitaryKVCacheCoordinator 和 HybridKVCacheCoordinator 不同，
    supports arbitrary numbers of KV cache groups (including 0 groups).
    此类支持任意数量的 KV cache group（包括 0 个）。
    Does not implement any features related to prefix caching.
    不实现任何与前缀缓存相关的功能。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the
    适用于仅有一个 KV cache group 的模型。例如，模型中的所有
    case for models with only one KV cache type, e.g., all attention layers use
    attention layer 都使用 full attention，或者
    full attention or all attention layers use sliding window attention.
    都使用滑动窗口 attention。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when
        # 对于仅使用 Mamba 的模型，禁用前缀缓存时 block_size 会设为 max_model_len，
        # prefix caching is disabled, and hash_block_size validation is skipped.
        # 并跳过 hash_block_size 校验。
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    适用于包含多种 KV cache 类型，
    thus multiple kv cache groups.
    因而包含多个 KV cache group 的混合模型。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # hash_block_size：计算 block 哈希时使用的 block 大小。
        # The actual block size usually equals hash_block_size, but in cases where
        # 实际 block 大小通常与 hash_block_size 相同；若不同 KV cache group
        # different KV cache groups have different block sizes, the actual block size
        # 的 block 大小不同，实际大小可能是
        # can be a multiple of hash_block_size.
        # hash_block_size 的整数倍。
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        按 spec 类型对 KV cache group 分组，以便在查找缓存命中时
        during cache hit lookup.
        高效地批量处理。
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            # 尝试查找 spec 相同的已有分组。
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups."
        )

        # Put full attention first: its efficient left-to-right scan provides
        # 将 full attention 放在最前：它高效的从左到右扫描可以
        # a tighter initial bound, reducing work for subsequent groups.
        # 得到更严格的初始上界，从而减少后续分组的工作量。
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # The LCM of the block sizes of all attention types.
        # 所有 attention 类型的 block 大小的最小公倍数（LCM）。
        # The cache hit length must be a multiple of the LCM of the block sizes
        # 缓存命中长度必须是各 block 大小最小公倍数的整数倍，
        # to make sure the cache hit length is a multiple of the block size of
        # 以确保命中长度是
        # each attention type. Requiring this because we don't support partial
        # 每种 attention 类型 block 大小的倍数。由于目前不支持
        # block cache hit yet.
        # 部分 block 命中，因此需要此约束。
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.
        使用迭代不动点算法查找最长缓存命中前缀。

        Each attention type either accepts the current candidate length or
        每种 attention 类型要么接受当前候选长度，要么
        reduces it. If any type reduces the length, restart checks over all
        缩短该长度。任一类型缩短长度后，都要重新检查所有
        types. This converges because length monotonically decreases and is
        类型。由于长度单调递减且
        bounded below by 0.
        下界为 0，该过程最终会收敛。

        Args:
            block_hashes: The block hashes of the request.
            block_hashes：请求的 block 哈希列表。
            max_cache_hit_length: The maximum length of the cache hit.
            max_cache_hit_length：缓存命中的最大长度。

        Returns:
            A tuple containing:
            返回一个元组，包含：
                - A tuple of the cache hit blocks for each single type manager.
                - 每种类型管理器对应的缓存命中 block 元组；
                - The number of tokens of the longest cache hit.
                - 最长缓存命中的 token 数量。
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            if kv_cache_spec.block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, kv_cache_spec.block_size
            )

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        while True:
            curr_hit_length = hit_length

            for spec, group_ids, manager_cls in self.attention_groups:
                is_full_attn = isinstance(spec, FullAttentionSpec)

                # Full attention: reuse cached blocks (downward-closed property)
                    # Full attention：复用已缓存的 block（满足向下封闭性质）。
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if is_full_attn and cached_blocks is not None:
                    # For full attention, we only need to compute the cache hit
                    # 对 full attention，只需计算一次缓存命中长度。
                    # length once. Starting from the second iteration, if the
                    # 从第二轮开始，若其他 group 缩短了
                    # curr_hit_length is reduced by other groups, we can simply
                    # curr_hit_length，只需保留
                    # keep the first (curr_hit_length // block_size) blocks from
                    # 上一轮结果中的前 curr_hit_length // block_size 个 block，
                    # the last iteration.
                    # 无需重新扫描。
                    num_blocks = curr_hit_length // spec.block_size
                    curr_hit_length = num_blocks * spec.block_size
                    for group_id in group_ids:
                        blocks = hit_blocks_by_group[group_id]
                        assert blocks is not None
                        del blocks[num_blocks:]
                else:
                    hit_blocks = manager_cls.find_longest_cache_hit(
                        block_hashes=_get_block_hashes(spec),
                        max_length=curr_hit_length,
                        kv_cache_group_ids=group_ids,
                        block_pool=self.block_pool,
                        kv_cache_spec=spec,
                        use_eagle=self.use_eagle,
                        alignment_tokens=self.lcm_block_size,
                    )
                    curr_hit_length = len(hit_blocks[0]) * spec.block_size
                    for group_id, blocks in zip(group_ids, hit_blocks):
                        hit_blocks_by_group[group_id] = blocks

            if curr_hit_length < hit_length:
                hit_length = curr_hit_length
            else:
                break

        return tuple(
            blocks if blocks is not None else [] for blocks in hit_blocks_by_group
        ), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
