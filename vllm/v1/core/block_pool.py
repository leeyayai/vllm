# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    保存用于前缀缓存的 block，将哈希直接映射到一个或多个 block
    from hash directly to a block or multiple blocks
    （即 `{block_hash: KVCacheBlocks}`）。
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
    - 通常一个 block_hash 只对应一个 KVCacheBlock，此时 KVCacheBlocks
        would simply be a KVCacheBlock.
      实际上就是一个 KVCacheBlock。
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}
    - 如果对应多个 block，KVCacheBlocks 则是 `{block_id: KVCacheBlock}` 字典。

    A cached block is a full block with a block hash that can be used
    缓存 block 是带有 block 哈希的完整 block，可用于
    for prefix caching.
    前缀缓存。
    The cached block may be used by running requests or in the
    缓存 block 可能正被运行中的请求使用，也可能位于
    free_block_queue that could potentially be evicted.
    free_block_queue 中等待后续淘汰。

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    注意 #1：当前不会对缓存中的 block 去重。  # 不去重
    meaning that if a block becomes full and is cached, we don't check
    block 填满并缓存时，不会检查缓存中是否已有
    if there is already an identical block in the cache. This is because
    相同 block。这是因为我们需要确保已分配的 block ID 不发生变化，
    we want to make sure the allocated block IDs won't change so that
    从而使
    block tables are append-only.
    block table 只追加、不修改已有内容。  # 这是为了确保已分配的 block ID 不发生变化，
    NOTE #2: The union type is introduced in order to reduce GC costs
    注意 #2：引入联合类型是为了降低
    from the inner dict.
    内部字典带来的垃圾回收开销。
    """

    # BlockHashToBlockMap维护一个从“Block 哈希”到“实际 KV Cache Block”的映射字典，用于实现前缀缓存的复用。
    # 如果多个请求有相同的 Prompt 前缀，我们不需要重复计算这些前缀的 KV Cache。
    # 这个类就是用来记住：“这个哈希值对应的 KV Cache Block 在哪里”。

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}
    # 联合类型
    # 当 block_hash 只对应一个 KVCacheBlock 时，KVCacheBlocks 就是一个 KVCacheBlock。-- 绝大多数情况
    # 当对应多个 block 时，KVCacheBlocks 就是一个 `{block_id: KVCacheBlock}` 字典。 -- 相同前缀的多个 Block，两个请求同时进入缓存时，会对应两个不同的 block_id，且发现内容相同，不做去重

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        获取任意一个匹配的 Block
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values())) # 随机/任意返回字典中的第一个 Block
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        将 KVCacheBlock 插入缓存。
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            # 键不存在时，将单个 block 与该键关联。
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # 如果已有相同键的 block，则将原 block
            # and the new block into a dict
            # 与新 block 合并到字典中。
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            # 如果值已经是字典，则直接插入该 block。
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        检查缓存中是否存在 block_hash，并从中弹出指定 block_id。
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            # 缓存中不存在该 block_hash。
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # TODO（Jialin）：如果找到该键，block_id 理应始终存在
        # in blocks. We currently keep the original behaviour for safety.
        # 于 blocks 中。为安全起见，目前保留原有行为。
        #
        # Will add block_id == blocks.block_id assertion and
        # 后续会增加 block_id == blocks.block_id 断言，
        # use del blocks[block_id] instead as followup.
        # 并改用 del blocks[block_id]。
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # 如果单个 block 的 ID 不匹配，应将该
            # block back (it should happen rarely)
            # block 放回缓存（这种情况应该很少发生）。
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # 尝试从 block 字典中弹出 block_id；如果字典中仍
            # contain blocks, put back to the cache.
            # 有其他 block，则将字典放回缓存。
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    管理 KVCacheBlock 的 BlockPool。
    It provides methods to allocate, free and cache the kv cache blocks. The
    它提供 KV cache block 的分配、释放和缓存方法。free_block_queue
    free_block_queue stores the free blocks in eviction order to enable
    按淘汰顺序保存空闲 block，以支持分配、释放和缓存淘汰。
    allocation, free, and cache eviction. The cached_block_hash_to_block
    cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    维护 block 哈希到缓存 block 的映射，以便
    by their block hash.
    根据哈希查找缓存 block。

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        num_gpu_blocks：池中的 block 数量。
        enable_caching: Whether to enable prefix caching.
        enable_caching：是否启用前缀缓存。
        hash_block_size: The block size of which the block hashes are computed.
        hash_block_size：计算 block 哈希时使用的 block 大小。
            The actual block size usually equals hash_block_size, but in cases
            实际 block 大小通常与 hash_block_size 相同；若不同 KV cache group
            where different KV cache groups have different block sizes, the
            的 block 大小不同，实际大小可能是
            actual block size can be a multiple of hash_block_size.
            hash_block_size 的整数倍。
        enable_kv_cache_events: Whether to enable kv cache events.
        enable_kv_cache_events：是否启用 KV cache 事件。
        metrics_collector: Optional metrics collector for tracking block residency.
        metrics_collector：可选的指标收集器，用于跟踪 block 驻留情况。
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        # 所有 KV cache block。
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # 空闲 block 队列以双向链表形式组织并操作 block，
        # list of free blocks (including eviction candidates when caching is
        # （启用缓存时也包括可淘汰候选项），
        # enabled).
        # 以此管理空闲 block。
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        # 用于查找 block 的缓存。
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # 作为占位 block，block_id 为 0。
        # The ref_cnt of null_block is not maintained, needs special care to
        # null_block 的 ref_cnt 不会被维护，因此必须特别注意，
        # avoid freeing it.
        # 避免释放它。
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        按 block 哈希获取 `kv_cache_group_ids` 中各 group 对应的缓存 block；
        `kv_cache_group_ids`, or None if cache miss for any group.
        任一 group 未命中时返回 None。
        If there are duplicated blocks, we return the first block in the cache.
        存在重复 block 时，返回缓存中的第一个。

        Args:
            block_hash: The hash value of the block.
            block_hash：block 的哈希值。
            kv_cache_group_ids: The ids of the KV cache groups.
            kv_cache_group_ids：KV cache group 的 ID 列表。

        Returns:
            The cached blocks if exists, or None.
            若存在则返回缓存 block 列表，否则返回 None。
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        为前缀缓存保存一组完整 block。
        This function takes a list of blocks that will have their block hash
        此函数接收一组 block，更新其哈希元数据，
        metadata to be updated and cached. Given a request, it updates the
        并将其存入
        metadata for each block and caching it in the
        `cached_block_hash_to_block`。
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        Request 对象会在创建时以及追加新 token 时
        when it is created and when new tokens are appended.
        立即计算 block 哈希。

        Args:
            request: The request to cache the blocks.
            request：要缓存 block 的请求。
            blocks: All blocks in the request.
            blocks：该请求的所有 block。
            num_cached_blocks: The number of blocks that are already cached.
            num_cached_blocks：已缓存的 block 数量。
            num_full_blocks: The number of blocks that are full and should
            num_full_blocks：当前已填满、且应在本函数调用后缓存的
                be cached after this function.
                block 数量。
            block_size: Number of tokens in each block.
            block_size：每个 block 包含的 token 数量。
            kv_cache_group_id: The id of the KV cache group.
            kv_cache_group_id：KV cache group 的 ID。
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            # 常见情况。
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # block_size 是 hash_block_size 的整数倍。这会发生在
            # different KV cache groups have different block sizes.
            # 不同 KV cache group 的 block 大小时。
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # 使用原始 block_hashes（粒度为 hash_block_size），
            # the original block_hashes (at the granularity of hash_block_size).
            # 重新按 block_size 的粒度计算 block_hashes。
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # 启用滑动窗口注意力等稀疏 attention 时，某些 block 可能是 null block，
            # sliding window attention. We skip null blocks here.
            # 此处会跳过这些 null block。
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            # 更新完整 block 的元数据，并将其加入缓存。
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.
        从空闲 block 池中获取新 block。

        Note that we do not check block cache in this function.
        注意：此函数不会检查 block cache。

        Args:
            num_blocks: The number of blocks to allocate.
            num_blocks：要分配的 block 数量。

        Returns:
            A list of new block.
            新 block 列表。
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        # 为了只遍历一次列表，这里略微重复了一部分代码。
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        如果 block 已缓存在 `cached_block_hash_to_block` 中，则重置其哈希
        metadata and evict it from the cache.
        元数据并将其从缓存中淘汰。

        Args:
            block: The block to evict.
            block：要淘汰的 block。

        Returns:
            True if the block is evicted, False otherwise.
            若 block 被淘汰则返回 True，否则返回 False。
        """
        # Clean up metrics tracking first to prevent leaks
        # 先清理指标跟踪状态，以防止泄漏。
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            # block 没有哈希，无需淘汰。
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # 在 cached_block_hash_to_block 中找不到该 block，
            # eviction is not needed
            # 无需执行淘汰。
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # FIXME（Chen）：不确定此处应返回 `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # 还是 `(hash_value, group_id)`。目前这样处理没有问题，因为
            # we disable hybrid kv cache manager when kv cache event is
            # 启用 KV cache event 时会禁用混合 KV cache manager，
            # enabled, so there is only one group.
            # 因此只有一个 group。
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        touch 一个 block 会将其引用计数加 1，并可能将它从空闲队列中移除。
        the block from the free queue. This is used when a block is hit by
        当另一个请求命中相同前缀时，
        another request with the same prefix.
        会使用此方法。

        Args:
            blocks: A list of blocks to touch.
            blocks：要 touch 的 block 列表。
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # ref_cnt=0 表示该 block 位于空闲列表中（即为可淘汰
            # candidate), so remove it.
            # 候选项），因此将它移除。
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        释放一组 block。block 应按
        eviction priority, where the first block will be evicted first.
        淘汰优先级排序，队首的 block 会最先被淘汰。

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
            ordered_blocks：按淘汰优先级排列的待释放 block 列表。
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        # 将迭代器物化为列表，以便多次遍历。
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.
        根据 block ID 从前缀缓存中淘汰 block。

        only evicts blocks that are currently cached (have a hash). blocks
        仅淘汰当前已缓存（具有哈希）的 block。引用计数
        with ref_cnt > 0 are not freed from the block pool, only evicted
        大于 0 的 block 不会从 block 池释放，只会从前缀缓存哈希表中
        from the prefix cache hash table.
        移除。

        Args:
            block_ids: Set of block IDs to evict from cache.
            block_ids：要从缓存中淘汰的 block ID 集合。
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        重置前缀缓存。此函数可用于 RLHF 流程中，
        flows to invalid prefix caching after the weights are updated,
        在模型权重更新后使前缀缓存失效，
        or used for resetting prefix caching status for benchmarking.
        也可用于基准测试时重置前缀缓存状态。

        Returns:
            bool: True if the prefix cache is successfully reset,
            bool：前缀缓存重置成功时为 True，
            False otherwise.
            否则为 False。
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        # 移除所有哈希，确保后续不会命中旧 block。
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        # 清除所有 block 上的哈希。
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.
        获取池中的空闲 block 数量。

        Returns:
            The number of free blocks.
            空闲 block 数量。
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.
        获取 KV cache 使用率。

        Returns:
            The KV cache usage (between 0.0 and 1.0).
            KV cache 使用率（范围为 0.0 到 1.0）。
        """

        # Subtract 1 to account for null block.
        # 减去 null block 所占的一个 block。
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.
        原子性地取出所有事件并清空队列。

        Returns:
            A list of KV cache events.
            KV cache 事件列表。
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
