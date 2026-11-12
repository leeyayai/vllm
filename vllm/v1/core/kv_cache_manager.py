# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)

# block 元数据 → 池内空闲/哈希索引 → 请求级分配管理

# KVCacheManager 负责“请求需要哪些 KV Block”；BlockPool 负责“从哪里拿 Block、如何释放和复用”；
# KVCacheBlock 是一个 Block 的元数据对象；FreeKVCacheBlockQueue 管理空闲 Block；
# BlockHashToBlockMap 通过前缀哈希查找缓存 Block；
# KVCacheBlocks 则把 Manager 的内部结果包装起来，交给 Scheduler 使用

#                     Scheduler
#                         │
#                         │ 调用 get_computed_blocks()
#                         │       allocate_slots()
#                         │       free()
#                         ▼
#                  KVCacheManager
#                         │
#                         │ 管理请求的 Block 列表
#                         │ 通过 Coordinator 分组管理
#                         ▼
#               KV Cache Coordinator
#                         │
#                         │ 调用一个或多个
#                         ▼
#                SingleTypeKVCacheManager
#                         │
#              ┌──────────┴──────────┐
#              │                     │
#              ▼                     ▼
#          BlockPool          请求 Block Table
#              │
#       ┌──────┼────────────┐
#       │      │            │
#       ▼      ▼            ▼
#  KVCacheBlock  FreeQueue  BlockHashToBlockMap
#       │          │            │
#       └──────────┴────────────┘
#              管理同一批物理 Block

@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        获取请求已计算并缓存的 block。
        Note that the computed blocks must be full.
        注意：已计算的 block 必须是完整 block。

        Args:
            request: The request to get the computed blocks.
            request：要查询已计算 block 的请求。

        Returns:
            A tuple containing:
            返回一个元组，包含：
                - A list of blocks that are computed for the request.
                - 请求已计算 block 的列表。
                - The number of computed tokens.
                - 已计算 token 的数量。
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.
        为请求追加新 token，并分配相应的 KV slots。

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        block 布局如下：
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:
        缩写说明：

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        注意：若新 token 同时包含已验证和未验证的草稿 token，
        tokens, we only cache the verified tokens (by capping the number at
        只缓存已验证的 token（将数量限制在
        `request.num_tokens`).
        `request.num_tokens` 以内）。

        The allocation has three stages:
        分配过程分为三个阶段：
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
            新分配的 block 列表。
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # 异步加载 KV 数据时，可能没有新 token 需要计算，
        # compute while still allocating slots for externally computed tokens.
        # 但仍需为外部计算的 token 分配 slot。
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # 已计算 token 数量等于原有已计算 token 数
        # the new prefix caching hits
        # 加上新命中的前缀缓存 token 数。
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_need_slot = min(
            total_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # 释放 attention 计算中会跳过的 block，
        # (e.g., tokens outside the sliding window).
        # 例如滑动窗口之外的 token 对应的 block。
        # We can do this even if we cannot schedule this request due to
        # 即使因空闲 block 不足而无法调度该请求，
        # insufficient free blocks.
        # 仍可先执行此操作。
        # Should call this function before allocating new blocks to reduce
        # 应在分配新 block 前调用此方法，以减少
        # the number of evicted blocks.
        # 需要淘汰的 block 数量。
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            # 无法分配新 block。
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # 先将新命中的已计算 block 追加到请求当前持有的 block 列表，
            # avoid the case where the new blocks cannot be allocated.
            # 避免之后无法分配新 block 时留下不一致状态。
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        # P/D: delay caching blocks if we have to recv from
        # P/D：如果需要从远端接收 KV，则延迟缓存这些 block，
        # remote. Update state for locally cached blocks.
        # 但更新本地缓存 block 的状态。
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # 注意（woosuk）：希望缓存到 num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # + num_external_computed_tokens + num_new_tokens 的位置，但必须排除
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # “不可提交”的 token（例如之后可能被拒绝的草稿 token）。
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # 因此将数量限制为 `request.num_tokens`，确保只缓存
        # "finalized" tokens are cached.
        # 已最终确定的 token。
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        释放为该请求分配的 block。
        We free the blocks in reverse order so that the tail blocks are evicted
        按逆序释放 block，使尾部 block 在启用缓存时优先被淘汰，
        first when caching is enabled.
        从而保留更靠前的前缀 block。

        Args:
            request: The request to free the blocks.
            request：要释放其 block 的请求。
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        移除 `blocks` 中不再需要的 block，并用
        the removed blocks with null_block.
        null_block 替换被移除的位置。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            total_computed_tokens: The total number of computed tokens, including
            total_computed_tokens：已计算 token 总数，包括
                local computed tokens and external computed tokens.
                本地和外部计算的 token。
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.
        根据 block ID 从前缀缓存中淘汰 block。

        Args:
            block_ids: Set of block IDs to evict from cache.
            block_ids：要从缓存中淘汰的 block ID 集合。
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        重置前缀缓存。此函数可用于 RLHF 流程中，
        flows to invalidate prefix caching after the weights are updated,
        在权重更新后使前缀缓存失效，
        or used for resetting prefix caching status for benchmarking.
        也可用于基准测试时重置前缀缓存状态。

        Returns:
            bool: True if the prefix cache is successfully reset,
            bool：前缀缓存重置成功时为 True，
            False otherwise.
            否则为 False。
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.
        计算每个 KV cache group 中的公共前缀 block 数量。

        The function selects a running request and iterates through its blocks.
        此函数选择一个运行中请求并遍历其 block。
        A block is considered a common prefix block if ALL requests with
        如果所有持有
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        已分配 KV cache 的请求都共享某个 block，则该 block 被视为公共前缀 block
        in req_to_blocks).
        （即 ref_cnt 等于 req_to_blocks 中的请求条目数）。

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        注意（woosuk）：已分配 KV cache 的请求数**大于或等于**
        than or equal to** the number of requests scheduled in the current step.
        当前步骤中调度的请求数，
        This is because having allocated KV cache only indicates that:
        因为已分配 KV cache 仅表示：
        1. The request has not yet finished, and
        1. 请求尚未结束；
        2. The request holds its blocks unfreed.
        2. 请求持有的 block 尚未释放。

        While all scheduled requests must have allocated KV cache, the inverse
        虽然所有已调度请求都必定已分配 KV cache，但反过来
        is not necessarily true. There may be requests with allocated KV cache
        不一定成立。可能存在已分配 KV cache、
        that are not scheduled in the current step.
        但本轮未调度的请求。

        This can result in an edge case where the number of common prefix blocks
        因此可能出现一种边缘情况：所有已调度请求共享某个前缀，
        is 0, even though all scheduled requests share a common prefix. This
        但公共前缀 block 数仍为 0。这是因为
        occurs because there may be unscheduled requests that do not share the
        也可能存在本轮未调度、且不共享该前缀的请求。
        common prefix. Currently, this case cannot be easily detected, so the
        目前难以检测这种情况，因此
        function returns 0 in such cases.
        此时本函数返回 0。

        Args:
            running_request_id: The request ID of any running request, used to
            running_request_id：任一运行中请求的 ID，用于
                identify the common prefix blocks.
                识别公共前缀 block。

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            list[int]：每个 KV cache group 的公共前缀 block 数。
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.
        从 block pool 中取出 KV cache 事件。

        Returns:
            A list of KV cache events.
            KV cache 事件列表。
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request.
        获取请求的 block。
        """
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request.
        获取请求的 block ID。
        """
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.
        启用缓存时，缓存该请求的 block。

        Args:
            request: The request to cache the blocks.
            request：要缓存 block 的请求。
            num_computed_tokens: The number of computed tokens, including tokens
            num_computed_tokens：已计算 token 数量，包括
                that are already cached and tokens to be cached.
                已缓存和将要缓存的 token。
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        # 仅当 block 列表非空时才创建新的 KVCacheBlocks。
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
