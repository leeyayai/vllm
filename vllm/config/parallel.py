# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# odelConfig 决定了模型“怎么算”，CacheConfig 决定了“怎么存”，那么 
# ParallelConfig 就决定了“怎么分”——即如何将庞大的模型和并发请求切分到多个 GPU 甚至多台服务器上。

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self
# Self: 用于类型提示，允许方法返回当前类的实例（常见于链式调用或配置构建器模式）

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
# vLLM 的跨平台抽象层。用于判断当前是 NVIDIA、AMD、Intel 还是其他硬件，从而选择不同的通信后端。
from vllm.utils.network_utils import get_open_ports_list
# 分布式通信需要网络端口。这个工具用于自动寻找空闲端口，避免多实例启动时的端口冲突。
from vllm.utils.torch_utils import cuda_device_count_stateless
# 无状态地获取 GPU 数量，避免在配置阶段就意外初始化 CUDA 上下文（这会导致后续多进程 fork 时崩溃）。

if TYPE_CHECKING: # Lazy Loading) 与 Ray 依赖
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"] # 定义 MoE（混合专家模型） 中专家的放置策略。
# linear: 将连续的专家放在同一张卡上。
# round_robin: 轮询分配，让每张卡上的专家种类更均匀，有助于负载均衡。
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"] # 定义多卡/多机通信的执行器后端。
# ray: 使用 Ray 框架进行分布式调度（适合大规模集群）
# mp: 使用 Python 原生 multiprocessing（适合单机多卡）。
# uni: 单进程模式（所有逻辑在一个进程内，适合单卡或调试）。
# external_launcher: 外部启动器（如 Slurm、TorchRun），vLLM 只负责接管已启动的进程组。
DataParallelBackend = Literal["ray", "mp"] # 定义数据并行的通信后端。
# ray: 使用 Ray 框架进行数据并行（适合大规模集群）
# mp: 使用 Python 原生 multiprocessing（适合单机多卡）。
EPLBPolicyOption = Literal["default"] # 定义专家并行负载均衡的策略。
# default: 使用默认的负载均衡策略。
All2AllBackend = Literal[  # 定义 MoE 模型中 All-to-All 通信 的底层实现。这是 MoE 推理的性能瓶颈所在。
    "naive", # 原始实现，不支持并行。
    "pplx", # PPLX 实现，支持并行。
    "deepep_high_throughput", # DeepEP 高吞吐量实现。
    "deepep_low_latency", # DeepEP 低延迟实现。
    "allgather_reducescatter", # All-gather + Reduce-Scatter 实现。
    "flashinfer_all2allv", # FlashInfer All-to-All 实现。
]


@config
@dataclass
class EPLBConfig: # MoE（混合专家模型） 的专家并行负载均衡配置。
    #  MoE 模型在推理时，不同的 Token 会被路由到不同的专家，容易导致某些 GPU 上的专家过载。
    # EPLBConfig 允许 vLLM 动态监控负载（window_size），并在一定步数后（step_interval）重新排列专家，
    # 甚至引入冗余专家（num_redundant_experts）来缓解热点。
    """Configuration for Expert Parallel Load Balancing (EP)."""

    window_size: int = 1000
    """Window size for expert load recording."""
    step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism."""

    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """
    log_balancedness_interval: int = 1
    """
    Interval for logging the balancedness.
    """
    use_async: bool = False
    """
    Whether to use non-blocking EPLB.
    """

    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB)."""

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups.""" # 将模型按层切分到多卡。
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups.""" # 将模型按张量切分到多卡。
    prefill_context_parallel_size: int = 1
    """Number of prefill context parallel groups.""" # CP上下文并行，分为 Prefill 和 Decode 两种：将预填充上下文切分到多卡。
    # 当上下文极长时，将序列切分到多卡上并行计算，解决单卡显存无法容纳超长 KV Cache 的问题。
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    # MoE 专属逻辑：对于混合专家模型（MoE），DP 不仅仅用于增加吞吐，还用于切分专家（Expert Parallelism）。代码注释明确指出：MoE 层的专家会根据 tensor_parallel_size * data_parallel_size 的乘积来进行分片。
    # 这意味着在 MoE 场景下，增加 DP 既能提升并发能力，又能降低单卡显存压力。
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group,
    set only in SPMD mode.""" # 本地与全局 Rank 管理
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master.""" #配置数据并行节点间的通信地址和端口。
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    is provided explicitly to vllm serve."""
    # External LB（外部负载均衡）：适用于“一个 Pod 对应一个 Rank”的宽泛 EP（Expert Parallel）部署。
    # 此时 vLLM 内部不做请求分发，完全依赖外部的 Nginx/HAProxy 或 K8s Service 将请求路由到不同的 vLLM 实例。
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank.""" 
    # Hybrid LB（混合负载均衡）：结合了内部和外部负载均衡。在一个物理节点上运行一个 vLLM 实例（包含多个本地 DP Rank），vLLM 内部负责在本地 Rank 之间分发请求；
    # 而外部 LB 负责在多个物理节点（vLLM 实例）之间分发请求。这种模式可以最大化利用单机内的 NVLink 带宽，同时兼顾集群级的扩展性。
    # 高级负载均衡模式（云原生/K8s 场景）
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    enable_expert_parallel: bool = False #  MoE 专家放置与通信
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """Expert parallelism configuration."""
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers:\n
    - "linear": Experts are placed in a contiguous manner. For example, with 4
      experts and 2 ranks, rank 0 will have experts [0, 1] and rank 1 will have
      experts [2, 3].\n
    - "round_robin": Experts are placed in a round-robin manner. For example,
      with 4 experts and 2 ranks, rank 0 will have experts [0, 2] and rank 1
      will have experts [1, 3]. This strategy can help improve load balancing
      for grouped expert models with no redundant experts."""
    #  MoE 专家并行通信后端
    # linear 是连续放置，round_robin 是轮询放置（有助于负载均衡）。
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:

    - "naive": Naive all2all implementation using broadcasts\n
    - "allgather_reducescatter": All2all based on allgather and reducescatter\n
    - "pplx": Use pplx kernels\n
    - "deepep_high_throughput": Use deepep high-throughput kernels\n
    - "deepep_low_latency": Use deepep low-latency kernels\n
    - "flashinfer_all2allv": Use flashinfer alltoallv kernels for mnnvl"""
    # MoE 推理中最耗时的环节。vLLM 提供了多种后端（如 deepep_low_latency, flashinfer_all2allv），
    # 让用户根据集群网络拓扑（NVLink vs RDMA）选择最优解。

    max_parallel_loading_workers: int | None = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""
    # 在多个批次中依次加载模型时，并行加载工人的最大数量。当使用张量并行和大型模型时，避免RAM OOM

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    ubatch_size: int = 0
    """Number of ubatch size."""

    dbo_decode_token_threshold: int = 32
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch."""
    # 双批次重叠执行。这是一种极致的延迟隐藏技术，通过将计算和通信（或不同阶段的计算）重叠，进一步榨干 GPU 性能。
    # 阈值参数用于控制何时触发微批次（Micro-batching）。
    # 仅包含解码的批的双批重叠阈值。如果请求中的令牌数量大于此阈值，则将使用微批处理。否则，请求将被单独处理
    dbo_prefill_token_threshold: int = 512  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch."""
    # 包含一个或多个前置文件的批的双批重叠阈值。如果请求中的令牌数量大于此阈值，则将使用微批处理。否则，请求将在单个批处理中处理。

    disable_nccl_for_dp_synchronization: bool = Field(default=None)
    """Forces the dp synchronization logic in vllm/v1/worker/dp_utils.py 
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    """

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""

    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    Note that tpu only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    master_port: int = 29501
    """distributed master port for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    node_rank: int = 0
    """distributed node rank for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    nnodes: int = 1
    """num of nodes for multi-node distributed 
    inference when distributed_executor_backend is mp."""

    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    decode_context_parallel_size: int = 1
    """Number of decode context parallel groups, because the world size does
    not change by dcp, it simply reuse the GPUs of TP group, and tp_size
    needs to be divisible by dcp_size."""

    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.

    """
    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP or PCP.
    For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
        and `total_cp_world_size = pcp_world_size * dcp_world_size`.
    store interleave_size tokens on total_cp_rank i,
    then store next interleave_size tokens on total_cp_rank i+1.
    Interleave_size=1: token-level alignment, where token `i` is stored on
        total_cp_rank `i % total_cp_world_size`.
    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks. Tokens are then stored
        in (rank i+1, block j) only after (rank i, block j) is fully occupied.
    Block_size should be greater than or equal to cp_kv_cache_interleave_size.
    Block_size should be divisible by cp_kv_cache_interleave_size.
    """

    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""

    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self: # 配置合法性校验 (Validation)
        if self._api_process_rank >= self._api_process_count: # API 进程 Rank 校验
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        if self.data_parallel_size_local > self.data_parallel_size: #  2. 本地 DP 组大小不能大于全局 DP 大小
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        if self.data_parallel_size <= 1 and self.data_parallel_external_lb: # 3. 外部负载均衡模式只能在 DP 大于 1 时使用
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if self.enable_eplb:
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        return self

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else self.ubatch_size

    # 高并发启动多个 vLLM 实例时，多个进程可能会同时探测到同一个空闲端口，导致 EADDRINUSE（端口被占用）错误。
    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        # get_next_dp_init_port 维护了一个端口列表，每次初始化新的进程组（Process Group）时弹出一个新端口，避免内部冲突。
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def stateless_init_dp_group(self) -> ProcessGroup:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        # stateless_init_dp_group 在捕获到端口占用异常时，会自动递增端口号并重试最多 5 次，极大提升了生产环境下的启动稳定性。
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend=current_platform.dist_backend,
                )
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism with DeepEP All2All ops, replicated
    # tokens results in useless duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    # Not needed for pplx-kernels as it can handle duplicate input tokens.
    @property
    def use_sequence_parallel_moe(self) -> bool:
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "naive",
                "deepep_high_throughput",
                "deepep_low_latency",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    @property
    def node_rank_within_dp(self) -> int:
        return self.node_rank % self.nnodes_within_dp

    @property
    def nnodes_within_dp(self) -> int:
        if self.nnodes == 1:
            return 1
        data_parallel_node_size = (
            self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    @property
    def local_world_size(self) -> int:
        return self.world_size // self.nnodes_within_dp

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self): # 计算哈希
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None: 
        # 负责处理环境变量兼容、计算全局拓扑（World Size）、解析数据并行（DP）参数，
        # 并智能决策底层应该使用哪种分布式执行器（Ray、Multiprocessing 还是单进程）
        # Set all2all_backend from env var if not specified, with deprecation warning
        if envs.is_set("VLLM_ALL2ALL_BACKEND"):
            logger.warning_once(
                "VLLM_ALL2ALL_BACKEND environment variable is deprecated and "
                "will be removed in v0.15.0. Please use the "
                "--all2all-backend command-line argument instead."
            )
            self.all2all_backend = envs.VLLM_ALL2ALL_BACKEND

        # Continue with the rest of the initialization
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self._data_parallel_master_port_list:
                self._data_parallel_master_port_list = get_open_ports_list(5)
            self.data_parallel_master_port = self._data_parallel_master_port_list.pop()

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and cuda_device_count_stateless() < self.world_size
            ):
                gpu_count = cuda_device_count_stateless()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from vllm.v1.executor import Executor

        # Enable batch invariance settings if requested
        if vllm_is_batch_invariant():
            self.disable_custom_all_reduce = True

        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            from vllm.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.nnodes > 1:
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce since we are running on multi-node."
            )
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        return self
