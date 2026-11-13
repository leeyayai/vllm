# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager # 结构化输出管理器
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder # 是否是编码-解码模型

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        # 在 Multi-Engine 场景中，可能有多个 Engine 协同处理请求。
        # 当一个请求完成时，需要让相关组件知道：
        # - 哪个请求已经结束？
        # - 哪个请求可以释放相关资源？
        # - 哪个请求不再需要继续追踪？
        # 通过集合保存已经完成的 request ID，可以高效地进行生命周期追踪。
        # 注意：这并不意味着所有请求都会使用这个字典。只有 include_finished_set=True 才会创建它。
        self.finished_req_ids_dict: dict[int, set[str]] | None = (   # 已完成请求ID字典
            defaultdict(set) if include_finished_set else None
        )
        # 为什么要记录？因为 vLLM 的调度是不断循环执行的：
        # Scheduler 需要知道上一步哪些请求参与了执行，以便在下一步处理调度状态、请求生命周期等逻辑。
        # 第 1 次调度
        #     ↓
        # GPU 执行
        #     ↓
        # 返回执行结果
        #     ↓
        # update_from_output()
        #     ↓
        # 第 2 次调度
        #     ↓
        # GPU 执行
        #     ↓
        # 返回执行结果
        #     ↓
        # ...

        self.prev_step_scheduled_req_ids: set[str] = set() # 上一步已调度请求ID集合

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs # 最大并发请求数
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens # 最大批量调度tokens数
        self.max_model_len = vllm_config.model_config.max_model_len # 最大模型长度
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )  # 是否启用KV缓存事件，用以记录的踪 KV Cache 相关变化的事件机制
        # 例如：
        #     - KV Cache Block 分配。
        #     - KV Cache Block 释放。
        #     - KV Cache Block 的相关状态变化。
        #     这类事件可以被外部监控或其他组件使用。

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks  # GPU KV Cache Block 的总数量
        assert num_gpu_blocks is not None and num_gpu_blocks > 0  # Scheduler 初始化时，KV Cache 的 Block 数量已经确定。
        # 初始化顺序：
        # 模型初始化
        #     ↓
        # GPU KV Cache 显存规划
        #     ↓
        # 确定 num_gpu_blocks
        #     ↓
        # 初始化 Scheduler
        #     ↓
        # Scheduler 获取 KV Cache 容量信息

        self.block_size = block_size  # 每个GPU Block的大小
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size  # 上下文并行CP
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size  # 上下文并行CP

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy  调度策略，策略枚举类SchedulingPolicy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.  维护两个队列
        self.waiting = create_request_queue(self.policy)  # 等待调度请求队列
        self.running: list[Request] = []  # 运行中的请求队列

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # 保存两个调度步骤之间已经完成的请求 ID。用以Scheduler 和 Worker 之间的状态同步
        # Driver / Scheduler
        #     |
        #     | request_001 已经完成
        #     |
        #     v
        # Worker
        #     |
        #     | 仍然可能保留 request_001 对应的状态
        #     |
        #     v
        # KV Cache / 其他缓存
        #Worker 收到通知后，可以清理对应请求的缓存状态。
            # 例如：
            # - 清理请求对应的模型运行状态。
            # - 释放相关的缓存资源。
            # - 清理需要按请求维护的辅助状态。
        # GPU KV Cache Block 的管理和 Worker 侧缓存状态清理是相关但不同的事情。
        # Scheduler 自己的 KV Cache Manager 也会管理 Block 的释放，Worker 还可能维护其他请求级状态。
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        # KV Connector负责连接不同 KV Cache 存储或传输机制，让请求能够异步加载、接收或者传输 KV Cache。
        # 在一些分布式推理场景下，可能需要从其他位置获取 KV Cache。从其他推理实例传输 KV Cache。
        # 从外部 KV Cache 存储系统加载 KV Cache。Prefill 和 Decode 分离部署时，在不同实例之间传输 KV Cache。
        # KV Cache 的异步接收是一个独立于普通模型计算的过程。请求可能已经进入调度体系，但它所需的 KV Cache 还没有完全准备好，
        # 因此不能简单地认为请求进入 Running 就一定可以马上执行模型计算。
        self.finished_recving_kv_req_ids: set[str] = set() # 已完成接收 KV Cache 的请求 ID
        # 失败接收 KV Cache 的请求 ID
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.  Encoder 相关初始化
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        # 因为 Encoder 侧有两个不同的问题：
        #     1. 当前调度步骤最多允许处理多少 Encoder 输入？
        #     2. 已经计算完成的 Encoder 结果，最多可以缓存多少？
        # 因为 vLLM 不只有纯文本 Decoder-only 模型，还支持多模态模型和 Encoder-Decoder 模型。
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )
        # encoder_compute_budget	Encoder 侧允许调度的计算预算
        # encoder_cache_size	Encoder Cache 可以使用的缓存容量

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget  # 保存 Encoder 计算预算
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )  
        # For encoder-decoder models, allocate the maximum number of tokens for Cross
        # Attn blocks, as for Whisper its input is always padded to the maximum length.
        # TODO (NickLucche): Generalize to models with variable-length encoder inputs.
        self._num_encoder_max_input_tokens = (
            MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(vllm_config.model_config)
        )   # 考虑的encoder-decoder模型，暂放

        # 投机解码配置
        speculative_config = vllm_config.speculative_config
        self.use_eagle = False  
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:  # 是否开启投机解码
            self.num_spec_tokens = speculative_config.num_speculative_tokens  # 投机解码的 token 数量
            if speculative_config.use_eagle():  # 是否开启投机解码的 Eagle 功能
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager. 初始化 KV Cache 管理器
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config, # KV Cache 配置
            max_model_len=self.max_model_len, # 最大模型输入长度
            enable_caching=self.cache_config.enable_prefix_caching, # 是否开启前缀缓存
            use_eagle=self.use_eagle, # 是否开启投机解码的 Eagle 功能
            log_stats=self.log_stats, # 是否开启统计信息志
            enable_kv_cache_events=self.enable_kv_cache_events, # 是否开启 KV Cache 事件
            dcp_world_size=self.dcp_world_size, # 数据并行世界大小
            pcp_world_size=self.pcp_world_size, # 模型并行世界大小
            hash_block_size=self.block_size, # KV Cache 哈希块大小
            metrics_collector=self.kv_metrics_collector, # KV Cache 统计信息收集器
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1 # 是否开启pp并行
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER  # 是否开启 v2 模型运行器  -- 
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts: # 是否开启返回MoE路由专家
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            ) # 目前不支持 Prefill Context Parallel 和 Decode Context Parallel

            self.routed_experts_reader = RoutedExpertsReader.create()
            # 创建专家路由信息读取器，负责读取模型执行过程中产生的专家路由相关信息 - 采集信息用于监控

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            ) # 至少需要一个 KV Cache 组
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // len(kv_cache_config.kv_cache_groups) + 1
            ) * self.block_size    # 计算最大 KV Cache 哈希块数

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                model_config=self.vllm_config.model_config,
                instance_id=self.vllm_config.instance_id,
            ) # 为专家路由信息读取器分配缓冲区，用于存储专家路由信息

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # 调度算法说明：
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Scheduler 不会把请求划分为“解码阶段”或“预填充阶段”。
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # num_tokens_with_spec 表示含推测 token 的 token 总数，计算方式为：
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # prompt_token_ids、output_token_ids 和 spec_token_ids 三者长度之和。
        # At each step, the scheduler tries to assign tokens to the requests
        # 每一步中，Scheduler 都会为请求分配 token，
        # so that each request's num_computed_tokens can catch up its
        # 使每个请求的 num_computed_tokens 逐步追上其
        # num_tokens_with_spec. This is general enough to cover
        # num_tokens_with_spec。该方法可以统一处理
        # chunked prefills, prefix caching, speculative decoding,
        # 分块预填充、前缀缓存和推测解码，
        # and the "jump decoding" optimization in the future.
        # 以及未来可能加入的“跳跃解码”优化。

        scheduled_new_reqs: list[Request] = []  # 本轮从 Waiting 中选出来的新请求
        scheduled_resumed_reqs: list[Request] = []   # 被抢占后重新恢复的请求
        scheduled_running_reqs: list[Request] = [] # 本轮本轮继续执行的 Running 请求
        preempted_reqs: list[Request] = []  # 本轮被抢占的请求

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}  	#请求和本轮分配的 KV Cache blocks 的对应关系，每个请求本轮被安排计算多少 KV Cache 哈希块
        num_scheduled_tokens: dict[str, int] = {}  # 请求和本轮分配的 token 数量对应关系，每个请求本轮被安排计算多少 Token
        token_budget = self.max_num_scheduled_tokens  # 本轮可分配的 token 数量预算
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens  # 本轮可分配的 encoder 输入 token 数量预算，防止 Encoder 输入占用过多计算资源
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {} #本轮使用的推测解码 Token，提供给 Speculative Decoding 流程

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests. 先处理 Running 请求
        # Running 队列里既可能有正在 Prefill 的请求，也可能有正在 Decode 的请求，还可能有处于 Chunked Prefill 过程中的请求
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # ------------------是否跳过当前请求------------------------------------
            if (
                request.num_output_placeholders > 0  # 说明当前请求存在 Output Placeholder。
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.  # 异步调度下，避免多调度一步
                and request.num_computed_tokens + 2 - request.num_output_placeholders # + 2 和 - num_output_placeholders 是结合异步调度。
                >= request.num_prompt_tokens + request.max_tokens # 即使上一轮预留的 Draft Token 全部被拒绝，当前请求是否也已经达到最大生成长度？
            ): 
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue   # 当前请求不再参与本轮调度，跳过当前请求

            # ----------------计算本轮当前请求可分配的 token 数量--------------------
            num_new_tokens = (
                request.num_tokens_with_spec  # 当前请求的 Token 序列长度，包含已经纳入请求状态的推测 Token。
                + request.num_output_placeholders
                - request.num_computed_tokens 
            )  # 计算当前请求本轮可分配的 token 数量
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold  # 限制单个请求的 Long Prefill 长度
            num_new_tokens = min(num_new_tokens, token_budget) # 计算本轮当前请求可分配的 token 数量

            #  --------------限制不能超过模型最大长度-------------------------------

            # Make sure the input position does not exceed the max model len.
            # 确保输入位置不超过模型允许的最大长度。
            # This is necessary when using spec decoding.
            # 使用推测解码时必须执行此限制。
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            #  --------------调度 Encoder 输入：Encoder-Decoder 模型 -- 暂放 -------------------------------

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget # encoder的输入限制 encoder budget
            if request.has_encoder_inputs: # 判断请求是否包含 Encoder 输入
                ( 
                    encoder_inputs_to_schedule,  # 本轮需要调度的 Encoder 输入
                    num_new_tokens,   # 经过 Encoder 调度约束后调整的 Decoder Token 数量
                    new_encoder_compute_budget,  # 更新后的 Encoder 计算预算
                    external_load_encoder_input,  # 需要从外部加载的 Encoder 输入索引
                ) = self._try_schedule_encoder_inputs(
                    request,  # 当前正在调度的请求
                    request.num_computed_tokens,  # 当前请求已经计算的位置
                    num_new_tokens,  # 本轮计划调度的 Token 数量
                    encoder_compute_budget,  # Encoder 剩余计算预算
                    shift_computed_tokens=1 if self.use_eagle else 0,  # 是否需要对计算位置进行偏移  -- use_eagle投机解码的eagle模式
                )

            if num_new_tokens == 0: # 无可分配资源
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue # 跳过  -- 不是break -- 因为上面三种原因--PP暂放-- encoder预算不足导致不能调度，依旧有token_budge预算可以调度下一轮

            #  -------------- KV Cache Block 分配与请求抢占（Preemption） -------------------------------
            
            # Schedule newly needed KV blocks for the request.
            # 为该请求分配本轮新增 token 所需的 KV block。   --  以每个请求为基础，计算num_new_tokens，分配对应的KV block数量
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens, # 	本轮需要调度的 Token 数量
                        num_lookahead_tokens=self.num_lookahead_tokens,  # Lookahead Token 数量，用于 Speculative Decoding 等场景
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    #------------------------KV Cache 分配失败，抢占运行中请求以腾出空间--------------------
                    # The request cannot be scheduled.
                    # 当前请求暂时无法调度。
                    # Preempt the lowest-priority request.
                    # 抢占优先级最低的运行中请求以腾出空间。
                    if self.policy == SchedulingPolicy.PRIORITY:  # 优先级调度策略，默认是时间
                        preempted_req = max(  # 在 PRIORITY 策略下，vLLM 的优先级数值越小，优先级越高。
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),  # 到达时间越晚，数值通常越大，也越可能被选中抢占
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:  # 如果被抢占的请求在运行中队列中 -- 
                            scheduled_running_reqs.remove(preempted_req)  # 从本轮已调度请求中移除
                            token_budget += num_scheduled_tokens[    # 恢复 Token Budget ， num_scheduled_tokens: dict[str, int] = {} 
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id) # req_to_new_blocks: dict[str, KVCacheBlocks] = {}
                            num_scheduled_tokens.pop(preempted_req.request_id) # 移除 Token 调度数量记录 num_scheduled_tokens: dict[str, int] = {}
                            scheduled_spec_decode_tokens.pop(  # 移除 Speculative Decoding 记录 scheduled_spec_decode_tokens: dict[str, int] = {}
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop( # 移除 Encoder 输入记录 scheduled_encoder_inputs: dict[str, list[int]] = {}
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i) # 获取被抢占请求的 Encoder 输入的 Embedding 数量
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1 # 索引减1，因为被抢占的请求已被移除，所以需要调整索引，保持连续
                    else: # 先进先出队列
                        preempted_req = self.running.pop()

                    # ------------------------执行 Preemption------------------------------------------
                    # 明确：waiting队列里都是需要从头prefill的，对于chunkprefill的，这块是如何设计的
                    self._preempt_request(preempted_req, scheduled_timestamp)  # 释放资源、重置计算进度，并把请求放回等待队列。 
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:  # while True 退出条件，自己抢占自己，没法调度
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:  # 如果当前请求无法调度，上面自己抢占自己的退出形式
                # Cannot schedule this request.
                break
            # ----------------------------分配成功，记录本轮调度结果---------------------------------
            # Schedule the request.
            scheduled_running_reqs.append(request)  #  kv分配成功后加入scheduled_running_reqs
            req_to_new_blocks[request.request_id] = new_blocks # 本轮为请求新分配的 KV Blocks req_to_new_blocks: dict[str, KVCacheBlocks] = {}
            num_scheduled_tokens[request.request_id] = num_new_tokens # 移除 Token 调度数量记录 num_scheduled_tokens: dict[str, int] = {}
            token_budget -= num_new_tokens
            req_index += 1  # 

            # ----------------------------投机调度记录--暂放---------------------------------
            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # ----------------------------Encoder 输入调度记录--暂放---------------------------------
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        # 创建一个临时队列，保存本轮调度过程中暂时跳过的 Waiting 请求，之后再把它们放回 Waiting 队列前面。
        skipped_waiting_requests = create_request_queue(self.policy) # 根据当前 Scheduler 使用的调度策略，创建对应类型的waiting队列。

        # ----------------------------调度 Waiting 请求---------------------------------
        # Next, schedule the WAITING requests.
        if not preempted_reqs:  # 如果当前请求没有被抢占，才调度等待队列中的请求。
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # ------------------------检查请求状态是否需要跳过---------------------------------

                # KVTransfer: skip request if still waiting for remote kvs.
                # 明确：waiting队列里都是需要从头prefill的，对于chunkprefill的，这块是如何设计的
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:  # 之前被抢占过
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()  # 删除并跳过当前请求，等待远程 KV 准备好
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation. FSM 是 Finite State Machine，也就是有限状态机，和 Structured Output 有关
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request # 拿到 Structured Output 请求对象
                    if structured_output_req and structured_output_req.grammar: # 检查 Grammar 是否已经准备好
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()   # 删除并跳过当前请求，等待 Grammar 准备好
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # LoRA 检查：为什么 scheduled_loras 达到上限就跳过？  ----  暂放
                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request) # 跳过
                    continue

                # ------------------------调度 Waiting 请求---------------------------------
                
                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.  
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.  检查在的本地 Prefix Cache 命中情况--- 前缀匹配
                    new_computed_blocks, num_new_local_computed_tokens = (  # 先检查本地有没有缓存
                        self.kv_cache_manager.get_computed_blocks(request)  # 传入的是request，不是prompt
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    # 检查在的远端 Prefix Cache 命中情况--- 前缀匹配
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    # 异步远端 KV 接收完成后，请求重新进入 Waiting 队列时，已经有了大于 0 的计算进度。
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None # 本轮需要计算的 Encoder 输入
                external_load_encoder_input = [] # 本轮需要从外部加载的 Encoder 输入索引
                new_encoder_compute_budget = encoder_compute_budget # 经过本轮 Encoder 输入调度后，新的计算预算

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self._update_connector_prefix_cache_stats(request) # 

                self.running.append(request)  # 
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)  # 
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        # 为 ECConnector 构建元数据。
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.
        抢占请求并将其放回等待队列。

        NOTE: The request should be popped from the running queue outside of this
        注意：调用此方法前，应先在方法外将请求从运行队列中弹出。
        method.
        """
        assert request.status == RequestStatus.RUNNING, ( 
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)  # 释放 Decoder KV Cache
        self.encoder_cache_manager.free(request)  # 释放 Encoder Cache
        request.status = RequestStatus.PREEMPTED  # 抢占后，状态设为 Preempted
        request.num_computed_tokens = 0  # 抢占后，重置已计算 token 数
        request.spec_token_ids.clear()
        request.num_preemptions += 1  # 抢占次数，增加1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        # 将请求放回等待队列。
        self.waiting.prepend_request(request)  # 先进先出队列，将请求放回队头

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # 在请求完成调度之后，再推进该请求的已计算 token 数。
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        # 1. 当前步骤的 scheduler_output 必须保留原始的调度 token 数，
        #    original number of scheduled tokens to determine input IDs.
        #    以便确定输入 ID。
        # 2. Advance the number of computed tokens here allowing us to
        # 2. 在此处推进已计算 token 数，可以让预填充请求
        #    schedule the prefill request again immediately in the next
        #    在下一次调度时立即继续处理，
        #    scheduling step.
        #    无需等待本轮输出回写。
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        # 3. 如果之后有 token（例如推测 token）被拒绝，
        #    computed tokens will be adjusted in update_from_output.
        #    update_from_output 会相应修正已计算 token 数。
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # 注意：_free_encoder_inputs 依赖 num_computed_tokens；该值
            # may be updated again in _update_from_output for speculative
            # 可能会在推测解码的 _update_from_output 中再次更新。
            # decoding. However, it is safe to call the method here because
            # 不过，此处调用仍然安全，因为
            # encoder inputs are always part of the prompt, not the output,
            # encoder 输入始终属于 prompt 而非模型输出，
            # and thus are unaffected by speculative decoding.
            # 因此不受推测解码影响。
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # 清空已完成请求 ID 集合。
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # 注意：这里不能调用 self.finished_req_ids.clear()，
        # it will also affect the scheduler output.
        # 否则也会清空 SchedulerOutput 中引用的集合。
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # 使用流水线并行（PP）时，Scheduler 会把采样出的 token 发回，
                # because there's no direct communication between the first-
                # 因为流水线首阶段和末阶段 Worker 之间没有直接通信。
                # stage worker and the last-stage worker. Otherwise, we don't
                # 其他情况下无需回传采样 token，
                # need to send the sampled tokens back because the model runner
                # 因为 Model Runner 会缓存这些 token。
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).

        确定当前步骤需要调度哪些 encoder 输入，并相应更新 `num_new_tokens`
        和 encoder token 预算。

        满足以下条件时会调度某个 encoder 输入：
        - 其输出 token 与本轮计算范围有重叠；
        - 尚未计算并存入 encoder cache；
        - ECConnector 的远端 encoder cache 中不存在该输入；
        - encoder token 预算足以处理该输入；
        - encoder cache 有足够空间存储结果。

        若因 cache 或预算限制无法调度某个 encoder 输入，本方法会调整
        `num_new_tokens`，仅调度到该 encoder 输入之前的 decoder token。

        注意，num_computed_tokens 同时包括本地缓存 block 和通过
        KVConnector 获取的外部缓存 block。
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # Check remote cache first
        # 首先检查远端缓存。
        if self.ec_connector is not None:
            remote_cache_has_item = self.ec_connector.has_caches(request)
        # NOTE: since scheduler operates on the request level (possibly with
        # 注意：Scheduler 按请求粒度工作（一个请求可能包含多个 encoder 输入），
        # multiple encoder inputs per request), we need to create temporary
        # 因此需要创建临时跟踪器，
        # trackers for accounting at the encoder input level.
        # 按 encoder 输入粒度进行预算统计。
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds

            # The encoder output is needed if the two ranges overlap:
            # 两个范围有重叠时才需要 encoder 输出：
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) 与
            # [start_pos, start_pos + num_encoder_tokens)
            # [start_pos, start_pos + num_encoder_tokens)。
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                # 本轮不需要该 encoder 输入。
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # encoder 输入已完成计算。
                # The calculation here is a bit different. We don't turn encoder
                # 这里的计算方式略有不同：encoder 输出不会转换成
                # output into tokens that get processed by the decoder and
                # 由 decoder 处理并计入
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # num_computed_tokens 的 token。start_pos 表示
                # the position where we need to ensure we calculate encoder
                # 需要确保计算 encoder 输入的位置；该值应始终为 0，
                # inputs. This should always be 0 to ensure we calculate encoder
                # 以保证在运行 decoder 前完成 encoder 输入计算。
                # inputs before running the decoder.  Once we've calculated some
                # 一旦已经计算过 decoder token（num_computed_tokens > 0），
                # decoder tokens (num_computed_tokens > 0), then we know we
                # 就可以确定 encoder 输入已经计算完毕，
                # already calculated encoder inputs and can skip here.
                # 因此可跳过这里的处理。
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # encoder 输入已完成计算，并存储在
                # in the decoder's KV cache.
                # decoder 的 KV cache 中。
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # 目前 encoder-decoder 模型尚未使用 encoder cache，
                # yet.
                # 因此仅处理非 encoder-decoder 模型。
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # 同一 encoder 输入已在本轮调度过，
                    # current step.
                    # 无需重复调度。
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # 该 encoder 输入已在之前步骤计算并缓存，
                    # previous step.
                    # 无需再次处理。
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # 如果不允许对 encoder 输入分块，就不能只调度
            # partially schedule a multimodal item. If the scheduled range would
            # 多模态项目的一部分。若当前调度范围只覆盖
            # only cover part of the mm input, roll back to before the mm item.
            # 部分多模态输入，则回退到该项目开始之前。
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                num_new_tokens = start_pos - num_computed_tokens
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # encoder cache 已满或 encoder 预算已耗尽。
                # NOTE(woosuk): We assume that the encoder input tokens should
                # 注意（woosuk）：假设 encoder 输入 token 必须整体处理，
                # be processed altogether, as the encoder usually uses
                # 因为 encoder 通常使用
                # bidirectional attention.
                # 双向注意力。
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # 只调度紧邻该 encoder 输入之前的 decoder token。
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # 由于前缀缓存，num_computed_tokens 可能大于
                    # than start_pos even though its encoder input is not
                    # start_pos，但对应的 encoder 输入仍不可用。
                    # available. In this case, we can't schedule any token for
                    # 此时本轮不能为该请求调度任何 token，
                    # the request in this step.
                    # 因此将新 token 数设为 0。
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # 计算当前调度范围内需要处理的 embedding 数量，
            # of scheduled encoder placholder tokens.
            # 范围由 encoder 占位 token 决定。
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # 当前范围内的 encoder 占位 token 不包含任何 embedding，
            # so we can skip the encoder input.
            # 因此可以跳过该 encoder 输入。
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and remote_cache_has_item[i]:
                mm_hashes_to_schedule.add(request.mm_features[i].identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # 收集本轮调度且使用结构化输出的请求 ID。
        # The corresponding rows of the bitmask will be in this order.
        # bitmask 中对应的行将按此顺序排列。
        # PERF: in case of chunked prefill,
        # 性能说明：分块预填充时，
        # request might not include any new tokens.
        # 请求可能没有新的 token。
        # Therefore, we might introduce some additional
        # 因此可能会额外执行一些
        # cycle to fill in the bitmask, which could be a big no-op.
        # 填充 bitmask 的迭代，而这些迭代可能完全没有实际工作。
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # 这些 block 中包含加载失败的外部计算 token。
            # load. Identify affected requests and adjust their computed token
            # 找出受影响的请求并调整其已计算 token 数，
            # count to trigger recomputation of the invalid blocks.
            # 以触发对无效 block 的重新计算。
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # 注意（woosuk）：num_scheduled_tokens 的长度可能达到 1K 甚至更多，
        # the below loop can be a performance bottleneck. We should do our best
        # 因此下面的循环可能成为性能瓶颈。应尽量
        # to avoid expensive operations inside the loop.
        # 避免在循环中执行开销较大的操作。
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                # 跳过 KV 加载失败且需要重新调度的请求。
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # 请求已经结束。这可能发生在
                # request is aborted while the model is executing it (e.g.,
                # 模型执行期间请求被中止时（例如
                # in pipeline parallelism).
                # 流水线并行场景）。
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # num_computed_tokens 表示本轮已处理的 token 数，
                # processed in the current step, considering scheduled
                # 其中要考虑本轮调度的 token
                # tokens and rejections. If some tokens are rejected,
                # 以及被拒绝的 token。若有 token 被拒绝，
                # num_computed_tokens is decreased by the number of rejected
                # num_computed_tokens 会按被拒绝 token 的数量
                # tokens.
                # 相应减少。
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # 异步调度时，num_output_placeholders 也包含
                # the scheduled spec tokens count and so is similarly adjusted.
                # 本轮调度的推测 token 数，因此也需要进行相同调整。
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            # 检查是否应停止生成，并更新请求状态。
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                # Pooling 请求一旦产生输出就立即结束。
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            if stopped:
                if self.vllm_config.model_config.enable_return_routed_experts:
                    kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
                    block_ids = kv_blocks.get_block_ids()[0]
                    num_tokens = request.num_tokens - 1

                    # compute slot mapping
                    # 计算 slot 映射。
                    block_ids_array = np.array(block_ids, dtype=np.int32)
                    num_blocks = len(block_ids)
                    block_size = self.block_size

                    # generate block offsets
                    # 生成 block 内的偏移量。
                    block_offsets = np.arange(0, block_size)

                    # compute slot mapping: slot = block_id * block_size + offset
                    # 计算 slot 映射：slot = block_id * block_size + offset。
                    slot_mapping = (
                        block_offsets.reshape((1, block_size))
                        + block_ids_array.reshape((num_blocks, 1)) * block_size
                    ).flatten()[:num_tokens]

                    routed_experts = self.routed_experts_reader.get_routed_experts(
                        indices=slot_mapping
                    )
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            # 如有需要，提取采样 token 的 logprobs。
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            # 获取此请求的 prompt logprobs。
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                # 为此请求添加 EngineCoreOutput。
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                # 不变量：EngineCore 不会返回不完整的预填充输出。
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        # 从运行队列和等待队列中移除已停止的请求。
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            # 这种情况很少见，预计不会影响性能。
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        # KV Connector：更新已完成 KV 传输的状态。
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        # 从 KV cache manager 收集 KV cache 事件。
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        # 从 connector 收集 KV cache 事件。
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        # 发布收集到的 KV cache 事件。
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # 为本轮有请求输出的所有客户端
        # outputs in this step.
        # 创建 EngineCoreOutputs。
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # 将上次发送输出之后结束的请求 ID
            # were sent.
            # 一并包含在结果中。
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                # 在该客户端的 EngineCoreOutputs 中设置已完成请求集合。
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            # 统计信息只返回给其中一个前端。
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # 即使本轮没有请求输出，也必须返回统计信息，
                # outputs this step.
                # 因此必要时创建一个空的 EngineCoreOutputs。
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # 追加生成的 token 并检查是否停止。注意，如果
        # a request is still being prefilled, we expect the model runner
        # 请求仍处于预填充阶段，Model Runner 应返回
        # to return empty token ids for the request.
        # 空的 token ID 列表。
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # 检查停止条件并更新请求状态。
            # This must be called before we make the EngineCoreOutput.
            # 必须先执行此步骤，再构造 EngineCoreOutput。
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        # 优化：集合为空时避免执行 list(set)。
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # 这里先转为 list(set)，以免在遍历集合时修改集合，
        # over it.
        # 导致迭代行为异常。
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # 对 Whisper 而言，一旦生成一个 token，
                # we know we're done with the encoder input. Cross Attention
                # 就可确认 encoder 输入已处理完毕。Cross Attention
                # KVs have been calculated and cached already.
                # 的 KV 已经计算并缓存。
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # encoder 输出已完成处理，并存储在
                # in the decoder's KV cache.
                # decoder 的 KV cache 中。
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                # 请求可能已经结束，跳过。
                continue

            # Add newly generated spec token ids to the request.
            # 将新生成的推测 token ID 加入请求。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                # 请求可能已经结束，跳过。
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # 将草稿 token 截断到本轮实际调度的推测 token 数量
            # (needed for chunked prefill case for example).
            # （例如分块预填充时需要这样处理）。
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            # 过滤不符合语法约束的推测 token。
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            # 将推测 token 补齐到原始数量。
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs).
        返回 (num_running_reqs, num_waiting_reqs)，即运行中和等待中的请求数。
        """
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str], finished_status: RequestStatus
    ) -> None:
        """Handles the finish signal from outside the scheduler.
        处理 Scheduler 外部发来的请求结束信号。

        For example, the API server can abort a request when the client
        例如，客户端断开连接时，API server 可以中止请求。
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def _update_connector_prefix_cache_stats(self, request: Request) -> None:
        if self.connector_prefix_cache_stats is None:
            return

        self.connector_prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=request.num_external_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

    def _make_connector_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if self.connector_prefix_cache_stats is None:
            return None
        stats = self.connector_prefix_cache_stats
        self.connector_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
