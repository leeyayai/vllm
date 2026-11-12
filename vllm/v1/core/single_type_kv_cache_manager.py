# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    管理一种特定 attention layer 的 KV cache 管理器抽象基类。
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        初始化 SingleTypeKVCacheManager。
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            kv_cache_spec：此管理器使用的 kv_cache_spec。
            block_pool: The block pool.
            block_pool：block 池。
            kv_cache_group_id: The id of the kv cache group of this manager.
            kv_cache_group_id：此管理器所属 KV cache group 的 ID。
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching

        # Mapping from request ID to blocks to track the blocks allocated
        # 请求 ID 到其 block 列表的映射，用于跟踪为各请求分配的 block，
        # for each request, so that we can free the blocks when the request
        # 以便请求结束时释放对应 block。
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}
        # {req_id: 该请求已缓存的 block 数量}
        # This is used to track the number of cached blocks for each request.
        # 用于跟踪每个请求的已缓存 block 数。
        # This is only used to track the RUNNING requests, we do not track the
        # 此映射只跟踪 RUNNING 请求，不记录
        # data for preempted ones.
        # 已被抢占请求的数据。
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
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
            total_computed_tokens: Include both local and external computed
            total_computed_tokens：本地和外部已计算 token 的总数。
                tokens.

        Returns:
            The number of blocks to allocate.
            要分配的 block 数量。
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # 快速路径：运行中的请求不会命中新前缀缓存。
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # 注意：推测解码可能已为请求的草稿 token 分配 block，
            # for draft tokens which are later rejected. In this case,
            # 之后这些 token 可能被拒绝。此时
            # num_required_blocks may be smaller than num_req_blocks.
            # num_required_blocks 可能小于 num_req_blocks。
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # 被 attention 窗口跳过的完整 block 数。
        # If nothing is skipped, this is 0.
        # 如果没有 block 被跳过，则为 0。
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # 需要为未跳过的后缀分配 block。如果窗口内仍有
        # local-computed blocks inside the window, they contribute to the
        # 本地计算的 block，它们也会计入
        # required capacity; otherwise, skipped blocks dominate.
        # 所需容量；否则由跳过的 block 数决定。
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # 在 `new_computed_blocks` 中，前 `num_skipped_blocks` 个 block
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # 会被跳过；其中可能有 `num_req_blocks` 个已经位于
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        # `req_to_blocks` 中，因此只需从 `new_computed_blocks` 跳过剩余部分。
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # 如果已计算 block 是待淘汰候选项（位于 free queue 且
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # ref_cnt == 0），被请求引用时会从 free queue 移除，
        # the allocated request, so we must count it in the free-capacity check.
        # 因此检查空闲容量时必须将其计入。
        num_evictable_blocks = sum(
            blk.ref_cnt == 0 and not blk.is_null
            for blk in new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        将新计算的 block 加入请求，分为以下步骤：
        1. Touch the computed blocks to make sure they won't be evicted.
        1. 标记已计算 block，确保其不会被淘汰。
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        1.5.（可选）滑动窗口会用 null block 填充被跳过的 block。
        2. Add the remaining computed blocks.
        2. 加入其余已计算 block。
        3. (Optional) For KV connectors, allocate new blocks for external computed
        3.（可选）使用 KV connector 时，为外部计算的
            tokens (if any).
            token（如有）分配新 block。

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

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # 快速路径：运行中的请求不会命中新前缀缓存。
            # It should not have any new computed blocks.
            # 因此不应有新的已计算 block。
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        # 这是一个新请求。
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks 大于 new_computed_blocks 数量时，
            # num_skipped_blocks > len(new_computed_blocks).
            # 所有新计算的 block 都可能被跳过。
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            # 部分外部计算的 token 也可能被跳过。
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        # 标记已计算 block，确保其不会被淘汰。
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        # 使用 null block 填充被跳过的 block。
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        # 加入其余已计算 block。
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # 所有命中缓存的 block（包括被跳过位置的 null block）都已缓存；
        # them so cache_blocks() will not try to re-cache blocks that already
        # 将它们标记出来，避免 cache_blocks() 再次缓存那些已经
        # have a block_hash set.
        # 设置了 block_hash 的 block。
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            # 为外部计算的 token 分配新 block。
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
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

        Returns:
            The new allocated blocks.
            新分配的 block。
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.
        缓存该请求的 block。

        Args:
            request: The request.
            request：目标请求。
            num_tokens: The total number of tokens that need to be cached
            num_tokens：需要缓存的 token 总数，
                (including tokens that are already cached).
                包括已经缓存的 token。
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.
        释放该请求的 block。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        # 如果请求在分配 block 前就被释放（中止），则默认使用空列表。
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # 按逆序释放 block，使尾部 block
        # freed first.
        # 优先被释放。
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        获取所有已分配 KV cache 的请求共有的
        KV cache.
        前缀 block 数量。

        Args:
            running_request_id: The request ID.
            running_request_id：请求 ID。

        Returns:
            The number of common prefix blocks for all requests with allocated
            所有已分配 KV cache 的请求共有的
            KV cache.
            前缀 block 数量。
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        获取长度不超过 `max_length` 的最长命中缓存前缀。
        `max_length`. The prefix should be a common prefix hit for all the
        该前缀必须同时命中
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        `kv_cache_group_ids` 中的所有 KV cache group。若没有命中，
        return an empty list.
        则返回空列表。
        If eagle is enabled, drop the last matched block to force recompute the
        启用 Eagle 时，会丢弃最后一个命中的 block，强制重新计算该 block，
        last block to get the required hidden states for eagle drafting head.
        以获取 Eagle drafting head 所需的 hidden state。
        Need to be customized for each attention type.
        不同 attention 类型需要分别实现此方法。

        Args:
            block_hashes: The block hashes of the request.
            block_hashes：请求的 block 哈希列表。
            max_length: The maximum length of the cache hit prefix.
            max_length：缓存命中前缀的最大长度。
            kv_cache_group_ids: The ids of the kv cache groups.
            kv_cache_group_ids：KV cache group 的 ID 列表。
            block_pool：block 池。
            kv_cache_spec：KV cache 规格。
            use_eagle：是否使用 Eagle。
            alignment_tokens：返回的缓存命中长度（token 数）必须是该值的倍数；
                默认应设为 block_size。
            dcp_world_size：decode context parallelism 的 world size。
            pcp_world_size：prefill context parallelism 的 world size。
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block
            返回各 KV cache group 的缓存 block 列表，被跳过的 block 用 null block
            for each kv cache group in `kv_cache_group_ids`.
            替换。
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            返回列表长度为 `len(kv_cache_group_ids)`，第 i 个元素
            element is a list of cached blocks for the i-th kv cache group
            是 `kv_cache_group_ids` 中第 i 个 KV cache group 的缓存 block 列表。
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            例如，block size 为 4、滑动窗口为 8 且 `len(kv_cache_group_ids) = 1` 时，
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            sliding window manager 应返回类似
            and sliding window 8 and len(kv_cache_group_ids) = 1.
            `([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)])` 的列表。
        """

        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        移除并释放 attention 计算不再需要的 block。
        The removed blocks should be replaced by null_block.
        被移除的位置应替换为 null_block。

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        此函数依赖 `get_num_skipped_tokens`；该方法需要根据
        differently for each attention type.
        attention 类型分别实现。

        Args:
            request_id: The request ID.
            request_id：请求 ID。
            total_computed_tokens: The total number of computed tokens, including
            total_computed_tokens：已计算 token 总数，包括
                local computed tokens and external computed tokens.
                本地和外部计算的 token。
        """
        # Remove the blocks that will be skipped during attention computation.
        # 移除 attention 计算中会被跳过的 block。
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # 这表示所有 token 都处于 attention 窗口内，
            # Thus we do not need to free any blocks outside attention window.
            # 因此无需释放窗口外的 block。
            # A typical case is full attention that we never free any token
            # 典型情况是 full attention：请求结束前
            # before the request is finished.
            # 不会释放任何 token。
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # `num_skipped_tokens` 可能包含尚未分配的 token，
        # (e.g., when the attention window moves into the external computed tokens
        # （例如 attention 窗口移入外部计算 token
        # range), so we must cap to the number of blocks that currently exist for
        # 的范围时），因此必须将其限制为当前
        # this request.
        # 为该请求实际存在的 block 数量。
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # 由于 block 索引从 0 开始，第 num_skipped_blocks 个 block
        # corresponds to index num_skipped_blocks - 1.
        # 对应的索引是 num_skipped_blocks - 1。
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # 如果该 block 已经是 null block，则此前的 block
                # should also have been set to null blocks by the previous calls
                # 也应已被之前调用
                # to this function.
                # 此函数时设为 null block。
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.
        获取 attention 计算中将被跳过的 token 数量。

        Args:
            num_computed_tokens: The number of tokens that have been computed.
            num_computed_tokens：已计算的 token 数量。

        Returns:
            The number of tokens that will be skipped for attention computation.
            attention 计算中将被跳过的 token 数量。
        """
        # The default behavior is to not skip any tokens.
        # 默认行为是不跳过任何 token。
        return 0


class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length is not compatible with eagle now"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: ChunkedLocalAttentionSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            + "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().get_num_blocks_to_allocate(
            request_id, num_tokens, new_computed_blocks, total_computed_tokens
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().allocate_new_blocks(request_id, num_tokens)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        return num_computed_tokens - 1


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
