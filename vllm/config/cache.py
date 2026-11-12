# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SkipValidation, field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes # 字节转换常量
from vllm.utils.mem_utils import format_gib, get_cpu_memory 
# format_gib（格式化显存大小显示）、get_cpu_memory（获取 CPU 内存）。这说明 CacheConfig 在初始化时会进行大量的显存/CPU内存计算和日志打印。

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
else:
    ParallelConfig = Any

logger = init_logger(__name__)

BlockSize = Literal[1, 8, 16, 32, 64, 128, 256] # 缓存块大小，单位是 Token 数量
CacheDType = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
] # 缓存数据类型，注意：这与模型权重的 dtype 是分开的
# 在长上下文推理时，KV Cache 往往比模型权重占用更多的显存。将其量化为 FP8 可以在几乎不损失精度的情况下，将 KV Cache 的显存占用减半。
MambaDType = Literal["auto", "float32", "float16"] # 定义 Mamba (SSM) 状态的数据类型型
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"] # 哈希算法，用于缓存键的前缀
KVOffloadingBackend = Literal["native", "lmcache"]# KV Cache Offloading 后端类型，支持 vLLM 原生 CPU offloading 和 lmcache。
# 当 GPU 显存不足时，可以将不活跃的 KV Cache 转移到 CPU 内存甚至 SSD 中。native 是 vLLM 自带的卸载机制，而 lmcache 是一个第三方的、更高效的 KV Cache 分布式缓存库。


@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. On CUDA devices,
    only block sizes up to 32 are supported.
    # 以令牌数量表示的连续缓存块的大小。在CUDA设备上，只支持最多32个块大小

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform.""" # 这是一个动态配置项，根据当前平台和用户输入进行设置。
    # 它的默认值是 None，这意味着用户可以不指定缓存块大小，而根据平台和模型自动选择。
    # 模型执行器使用的 GPU 内存比例。范围从 0 到 1，例如 0.5 表示使用 50% 的 GPU 内存。
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    # 每个 GPU 的 CPU 交换空间大小，单位是 GiB。
    # 用于模型执行器的GPU内存的比例，其范围从0到1。例如，0.5的值意味着50%的GPU内存利用率。如果未指定，将使用默认值0.9。这是每个实例的限制，并且只适用于当前vLLM实例。如果在同一GPU上运行另一个vLLM实例，这并不重要。
    # 例如，如果在同一个GPU上运行两个vLLM实例，则可以将每个实例的GPU内存利用率设置为0.5
    # 这个值用于在 GPU 显存不足时，将不活跃的 KV Cache 转移到 CPU 内存甚至 SSD 中。
    swap_space: float = Field(default=4, ge=0)
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    """
    # KV Cache 的存储数据类型。默认 "auto" 会使用模型权重相同的类型。支持在 CUDA 11.8+、ROCm、Intel Gaudi 上使用 FP8 格式（如 fp8_e4m3）来大幅降低显存占用
    is_attention_free: bool = False # 是否开启前缀缓存
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: bool = True # 是否开启前缀缓存
    """Whether to enable prefix caching."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256" # 前缀缓存使用的哈希算法。
    """Set the hash algorithm for prefix caching:\n
    - "sha256" uses Pickle for object serialization before hashing. This is the
    current default, as SHA256 is the most secure choice to avoid potential
    hash collisions.\n # 默认选项，最安全，避免哈希碰撞。
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
    serializes objects using canonical CBOR and hashes them with SHA-256.\n # 跨语言兼容，使用 CBOR 序列化。
    - "xxhash" uses Pickle serialization with xxHash (128-bit) for faster,
    non-cryptographic hashing. Requires the optional ``xxhash`` package.
    IMPORTANT: Use of a hashing algorithm that is not considered 
    cryptographically secure theoretically increases the risk of hash collisions,
    which can cause undefined behavior or even leak private information in
    multi-tenant environments. Even if collisions are still very unlikely, it is
    important to consider your security risk tolerance against the performance
    benefits before turning this on.\n # 速度极快，但非加密级，理论上存在哈希碰撞风险（可能导致多租户环境下的隐私泄露或异常行为
    - "xxhash_cbor" combines canonical CBOR serialization with xxHash for
    reproducible hashing. Requires the optional ``xxhash`` package."""
    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    # 卸载到 CPU 的空间（单位 GiB，默认 0）。相当于虚拟增加 GPU 显存。
    # 例如 24GB 显卡设置 10，可视为 34GB，从而加载原本显存不够的大模型。但这需要高速 CPU-GPU 互联，因为每次前向传播都会动态搬运数据。
    calculate_kv_scales: bool = False # 是否开启动态计算 KV � 缩放因子
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    cpu_kvcache_space_bytes: int | None = None # CPU KV 缓存空间（单位字节）
    """(CPU backend only) CPU key-value cache space."""
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache.
    Can be set only when prefix caching is enabled.
    Value must be a multiple of 8 to align with causal_conv1d kernel."""
    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""

    # Will be set after profiling.
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    """
    # 开启后，会跳过预填充，直接从缓存中获取。
    # 实验性功能。在某些 KV 共享架构（如 YOCO）中，部分层可以跳过 Prefill 阶段的 Token，以加速推理。

    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and vllm can automatically infer the kv cache size based on
    gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""
    #  允许用户以字节为单位精确指定 KV Cache 大小，优先级高于 gpu_memory_utilization，提供更细粒度的显存控制。

    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set, vLLM will
    enable KV cache offloading to CPU using the kv_offloading_backend."""

    kv_offloading_backend: KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache'.
    KV offloading is only activated when kv_offloading_size is set."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """ # 同modelConfig
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "swap_space",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "cpu_kvcache_space_bytes",
            "mamba_page_size_padded",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
        } # 忽略的配置项，因为它们不影响编译图的结构

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def metrics_info(self): # 为 Prometheus 监控系统提供配置字典
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}
    # 使用 Python 的 __dict__ 魔术方法，将当前 CacheConfig 对象的所有实例属性
    # （如 block_size, gpu_memory_utilization 等）提取出来，并将所有的值强制转换为字符串（str(value)）。

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: CacheDType) -> CacheDType:
        if cache_dtype.startswith("fp8"): # 校验 KV Cache 数据类型，并给出 FP8 相关的性能与精度警告。
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor."
            ) # 如果检测到用户启用了 FP8 格式（如 fp8_e4m3），它会打印一条 INFO 级别的日志。
        return cache_dtype
    # FP8 缓存能显著降低显存占用并提升推理速度，但它是一把“双刃剑”。如果模型在微调或量化时没有正确计算缩放因子（Scaling Factor），
    # FP8 会导致严重的精度下降（Accuracy Drop）。vLLM 通过这条日志，提醒开发者在享受性能红利的同时，注意潜在的精度风险。

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:  # 校验 CPU Swap 空间配置，防止因分配过多 Swap 导致系统 OOM（内存溢出）
        swap_space_bytes = math.ceil(self.swap_space * GiB_bytes)
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = swap_space_bytes * num_gpus_per_node

        msg = (
            f"{format_gib(cpu_memory_usage)} GiB out of the "
            f"{format_gib(total_cpu_memory)} GiB total CPU memory "
            "is allocated for the swap space."
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:  # 如果 Swap 空间占用超过 70%，则抛出异常
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory: # 打印 WARNING 警告，提示用户 Swap 空间可能过大，存在潜在风险
            logger.warning("Possibly too large swap space. %s", msg)
        # 当 GPU 显存不足时，vLLM 会将不活跃的 KV Cache 卸载（Offload）到 CPU 内存中。这个校验机制是 vLLM 的“安全网”，
        # 确保用户在配置 --swap-space 时，不会无意中耗尽宿主机的 CPU 内存，导致整个服务器崩溃。
