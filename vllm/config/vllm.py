# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 为整个vllm框架导入配置参数
# 为整个 vLLM 引擎引入所需的依赖项、定义类型别名，
# 并导入各个子模块的配置类（如模型配置、缓存配置、调度器配置等）。

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import is_dataclass, replace
from datetime import datetime
from enum import IntEnum
from functools import lru_cache  #lru_cache 用于缓存
from pathlib import Path  #Path 用于路径处理
from typing import TYPE_CHECKING, Any, TypeVar, get_args  #TYPE_CHECKING 用于类型检查时的条件导入

import torch
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
# vLLM 使用 pydantic 来进行严格的数据验证和配置管理。
# 通过 dataclass 和 Field，vLLM 可以确保传入的参数类型正确，并在参数缺失时提供默认值。

#vLLM 内部工具与环境变量导入，导入 vLLM 框架自身的工具函数。
import vllm.envs as envs #envs 用于读取环境变量
from vllm.config.speculative import EagleModelTypes
from vllm.logger import enable_trace_function_call, init_logger  #init_logger 用于初始化日志系统
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
# is_runai_obj_uri 用于判断是否使用了 Run:ai 等分布式资源调度平台。
from vllm.utils import random_uuid
from vllm.utils.hashing import safe_hash#safe_hash 用于生成哈希值（在 Prefix Caching 前缀缓存中非常关键）

# 核心子模块配置类导入
# vLLM 的模块化配置架构体现。vLLM 将极其复杂的引擎参数拆分成了多个独立的配置类
from .attention import AttentionConfig #管理注意力机制相关配置
from .cache import CacheConfig #管理 KV Cache 显存分配
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
# 编译配置类，用于管理模型的编译选项，如是否使用 CUDA、是否使用 CUDAGraph 等。
from .device import DeviceConfig
# 设备配置类，用于管理模型运行的硬件设备，如 GPU、CPU 等。
from .ec_transfer import ECTransferConfig
# EC 转移配置类，用于管理模型在不同设备之间的数据传输。
from .kv_events import KVEventsConfig
# KV 事件配置类，用于管理模型在不同设备之间的 KV 数据事件。
from .kv_transfer import KVTransferConfig
# KV 转移配置类，用于管理模型在不同设备之间的 KV 数据传输。
from .load import LoadConfig
# 加载配置类，用于管理模型的加载选项，如是否使用 CUDA、是否使用 CUDAGraph 等。
from .lora import LoRAConfig
# LoRA 配置类，用于管理 LoRA（Low-Rank Adaptation）相关的参数。
from .model import ModelConfig
# 模型配置类，用于管理模型的架构、参数、训练选项等。
from .observability import ObservabilityConfig
# 可观测性配置类，用于管理模型的监控、日志记录、指标收集等。    
from .parallel import ParallelConfig
# 并行配置类，用于管理模型的分布式训练、数据并行、模型并行等选项。
from .profiler import ProfilerConfig
# Profiler 配置类，用于管理模型的性能分析、调试等选项。
from .scheduler import SchedulerConfig
# 调度器配置类，用于管理模型的任务调度、资源分配等选项。
from .speculative import SpeculativeConfig
# 预计算配置类，用于管理模型的预计算选项，如是否使用预计算张量等。
from .structured_outputs import StructuredOutputsConfig
# 结构化输出配置类，用于管理模型的输出格式、输出类型等选项。
from .utils import SupportsHash, config
# 工具类，用于管理模型的哈希值、配置等。

if TYPE_CHECKING:  # 性能优化技巧
    # TYPE_CHECKING 是一个特殊的类型检查标志，用于在类型检查时导入模块，而不会在运行时实际导入它们。
    # TYPE_CHECKING 是一个在运行时永远为 False 的常量，但在 IDE 或静态类型检查工具（如 mypy）中为 True。
    from transformers import PretrainedConfig

    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    # 在类型检查时：导入 PretrainedConfig（HuggingFace 模型配置）、
    # QuantizationConfig（量化配置）等，让 IDE 能够正确提示代码。
else:
    PretrainedConfig = Any

    QuantizationConfig = Any

    KVCacheConfig = Any
    #运行时，将这些复杂的类直接赋值为 Any。
    # 这样可以避免在 vLLM 启动时导入庞大的 HuggingFace Transformers 库，从而显著加快 vLLM 的启动速度并减少内存占用。

logger = init_logger(__name__)


class OptimizationLevel(IntEnum): # 优化级别枚举
    """Optimization level enum."""
    # 定义了四个不同的优化等级，级别越高，编译和优化越彻底，但启动时的编译耗时也会相应增加。

    O0 = 0
    """O0 : No optimization. no compilation, no cudagraphs, no other
    optimization, just starting up immediately"""
    # 不进行任何优化，直接启动模型。不会使用 Torch Dynamo/Inductor 编译，也不会捕获 CUDA Graphs
    # 启动快
    O1 = 1
    """O1: Quick optimizations. Dynamo+Inductor compilation and Piecewise
    cudagraphs"""
    # 使用 Torch Dynamo + Inductor 进行图编译，并仅捕获“分段”（Piecewise）的 CUDA Graphs。
    O2 = 2
    """O2: Full optimizations. -O1 as well as Full and Piecewise cudagraphs."""
    #进一步捕获“完整”（Full）和“分段”（Piecewise）的 CUDA Graphs。
    # 这是生产环境推荐的最高性能模式，能最大化推理吞吐量，但启动时的编译和图捕获时间最长。
    O3 = 3
    """O3: Currently the same as -O2s."""  # 预留级别


IS_QUANTIZED = False
IS_DENSE = False
# The optimizations that depend on these properties currently set to False
# in all cases.
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# See https://github.com/vllm-project/vllm/issues/25689.

# 算子融合（Operator Fusion）条件判断函数。
# 多个小算子（如归一化、激活函数）合并为一个大算子，可以显著减少显存读写次数，从而大幅提升性能。
# 在什么情况下，vLLM 需要手动介入并开启这些融合优化。
def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    """Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""
    # enable_norm_fusion
    # 判断是否开启 RMSNorm（均方根归一化）的算子融合。
    

    return cfg.compilation_config.is_custom_op_enabled(
        "rms_norm"  # 判断是否启用了 RMSNorm 自定义算子
    ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")  # 判断是否启用了 FP8 量化自定义算子
    # 如果启用了 RMSNorm 或 FP8 量化自定义算子，就开启归一化算子融合。
    # 如果上述两个自定义算子都没有被激活，
    # 那么 vLLM 就会把这部分工作交给 PyTorch 的 Inductor 编译器去自动处理融合（otherwise Inductor handles fusion）。


def enable_act_fusion(cfg: "VllmConfig") -> bool:
    # enable_act_fusion
    # 判断是否开启 SiLU+Mul（SiLU 激活函数）的算子融合。

    """Enable if either SiLU+Mul or quant FP8 custom op is active;
    otherwise Inductor handles fusion."""
    return cfg.compilation_config.is_custom_op_enabled(
        "silu_and_mul" # 判断是否启用了 SiLU+Mul 自定义算子
    ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")  # 判断是否启用了 FP8 量化自定义算子
# 如果启用了 SiLU+Mul 或 FP8 量化自定义算子，就开启激活函数算子融合。
# 如果上述两个自定义算子都没有被激活，
# 那么 vLLM 就会把这部分工作交给 PyTorch 的 Inductor 编译器去自动处理融合（otherwise Inductor handles fusion）。

# CustomOp 优先：当 vLLM 针对特定硬件（如 NVIDIA GPU）提供了高度优化的底层 C++/CUDA 自定义算子（如手写的 FP8 量化、RMSNorm、SiLU+Mul）时，vLLM 会优先使用这些算子，并在编译配置中开启对应的融合。
# 其它芯片的vllm的插件也是一样的
# Inductor 兜底：如果没有使用这些特定的自定义算子，vLLM 不会让模型以低效的原始状态运行，而是将普通的 PyTorch 代码交给 Inductor 编译器，利用 Inductor 的自动图优化能力来完成算子融合。


# 通过 OPTIMIZATION_LEVEL_TO_CONFIG 将它们与 OptimizationLevel 枚举类进行了映射。
#在 vLLM 的编译管线中，算子融合（Operator Fusion）是提升推理性能的核心手段，它通过将多个小算子合并为一个大算子，减少显存读写次数（即“让数据少跑路”）。
# 这段代码正是通过控制 pass_config 中的各个开关，来决定在不同级别下开启哪些融合与优化策略。
OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": False, 
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
}
# eliminate_noops 是否剔除计算图中没有实际作用的空操作（No-ops），精简计算图
# fuse_norm_quant 是否将归一化（Normalization）和量化（Quantization）操作融合为一个算子，减少显存读写
# fuse_act_quant 是否将激活函数（Activation）和量化（Quantization）操作融合为一个算子，减少显存读写
# fuse_allreduce_rms 是否将 RMSNorm（均方根归一化）和 AllReduce（全局reduce）操作融合为一个算子，减少显存读写
# fuse_attn_quant 是否将注意力（Attention）和量化（Quantization）操作融合为一个算子，减少显存读写
    # 只有在当前模型是量化模型时，才会开启注意力机制与量化的融合。
# enable_sp 是否开启 Sp（Sparse）算子融合，减少显存占用
    # 只有在当前模型是密集模型时，才会开启 Sp 算子融合。
# fuse_gemm_comms 是否开启 Gemm（Matrix-Matrix Multiplication矩阵乘法与通信融合）算子融合，减少显存占用
    # 只有在当前模型是密集模型时，才会开启 Gemm 算子融合。
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "eliminate_noops": True,
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
}

OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


@config # config 装饰器用于注册配置类，使其可以被 vLLM 的配置管理系统识别和使用。
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# VllmConfig 类包含了所有与 vLLM 相关的配置项，包括模型配置、缓存配置、并行配置、调度器配置、设备配置、加载配置、注意力配置、LoRA 配置、推测配置、结构化输出配置、观察配置、量化配置、编译配置等。
# 类似Java的@data注解，简化了配置类的定义，使代码更简洁。
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """
    # 推理引擎的全局核心配置类 VllmConfig
    # 避免在函数调用时传递几十个零散的参数，框架将所有的配置项进行了模块化拆分，然后统一封装进这个 VllmConfig 数据类中。

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = Field(default=None) # 模型配置
    """Model configuration."""
    cache_config: CacheConfig = Field(default_factory=CacheConfig) # 缓存配置
    """Cache configuration."""
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig) # 并行配置
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = Field(  # 调度器配置
        default_factory=SchedulerConfig.default_factory,
    )
    """Scheduler configuration."""
    device_config: DeviceConfig = Field(default_factory=DeviceConfig) # 设备配置
    """Device configuration."""
    load_config: LoadConfig = Field(default_factory=LoadConfig) # 加载配置
    """Load configuration."""
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig) # 注意力配置
    """Attention configuration."""
    lora_config: LoRAConfig | None = None
    """LoRA configuration."""
    speculative_config: SpeculativeConfig | None = None # 投机采样（
    """Speculative decoding configuration."""
    structured_outputs_config: StructuredOutputsConfig = Field( # 结构化输出配置
        default_factory=StructuredOutputsConfig
    )
    """Structured outputs configuration."""
    observability_config: ObservabilityConfig = Field( # 观察配置，集成 Prometheus 等监控指标。
        default_factory=ObservabilityConfig
    )
    """Observability configuration."""
    quant_config: QuantizationConfig | None = None # 量化配置
    """Quantization configuration."""
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig) # 编译配置，编译与 CUDA Graph 配置
    # 对应上面的 OPTIMIZATION_LEVEL_TO_CONFIG 中的配置项，O0，O1，O2，O3
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, one can append compilation arguments via
    -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

    You can specify the full compilation config like so:
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig) #性能分析（Profiling）配置。用于抓取 GPU 底层的性能瓶颈。
    """Profiling configuration."""
    kv_transfer_config: KVTransferConfig | None = None # 分布式 KV 缓存传输配置，用于多节点推理或分离式架构（Prefill 和 Decode 分离）时的缓存同步。
    """The configurations for distributed KV cache transfer."""
    kv_events_config: KVEventsConfig | None = None # 分布式 KV 事件配置，用于发布 KV配置。，用于将 KV Cache 的分配/释放事件推送到外部系统。
    """The configurations for event publishing."""
    ec_transfer_config: ECTransferConfig | None = None # 分布式 EC 缓存传输配置，用于多节点推理或分离式架构（Prefill 和 Decode 分离）时的缓存同步。
    """The configurations for distributed EC cache transfer."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    # 附加配置。一个不透明的字典，主要用于测试、调试或第三方平台（Out-of-Tree）注册自定义配置。
    # 要求内容必须是可哈希的（hashable），因为它会被用来计算配置的唯一哈希值。
    instance_id: str = ""
    """The ID of the vLLM instance."""
    # vLLM 实例 ID。用于在分布式或多实例环境下唯一标识当前引擎进程。
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """The optimization level. These levels trade startup time cost for
    performance, with -O0 having the best startup time and -O3 having the best
    performance. -02 is used by defult. See  OptimizationLevel for full
    description."""

    def compute_hash(self) -> str:
        # 配置哈希计算。用于唯一标识所有影响计算图结构的配置项
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # 关键设计：Docstring 中特别警告，只有影响计算图结构（从输入 ID 到最终隐藏状态）的配置才应被包含在内。
        # 这主要用于 Prefix Caching（前缀缓存） 或 CUDA Graph 缓存，
        # 确保只要计算图结构不变，就可以安全复用之前的缓存。
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        from vllm import __version__

        vllm_factors.append(__version__)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.attention_config:
            vllm_factors.append(self.attention_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.structured_outputs_config:
            vllm_factors.append(self.structured_outputs_config.compute_hash())
        if self.profiler_config:
            vllm_factors.append(self.profiler_config.compute_hash())
        else:
            vllm_factors.append("None")
        vllm_factors.append(self.observability_config.compute_hash())
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.ec_transfer_config:
            vllm_factors.append(self.ec_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        # eagle.py和cudagraph.py中会调用这个函数来获取pad后的batch_size
        # Q：为什么需要这个函数？
        # A：为了在不同 batch size 下运行的 CUDA Graph 中，能够正确地分配资源和避免内存碎片。
        # Q：为什么要 pad batch size？
        # A：因为 CUDA Graph 只能在特定的 batch size 下运行。
        # Q: 为什么要查表？ -- 只固定了表里的 batch size 才能运行 CUDA Graph，只有这几种
        # A: 因为不同的 batch size 对应不同的 CUDA Graph 尺
        # Q: 传入模型的尺寸是flating的，不再pading的cudagraph在哪个阶段
        # A：vLLM 的输入是 Flattening（扁平化、非 Padding）的，但是 CUDA Graph 执行时，仍然可能需要对 Batch Size 进行 Padding。
        # Scheduler 调度 -Input Preparation-Flattening-CUDA Graph Shape 适配（根据 Graph 捕获规格，确定 Batch Size、Token 数等执行维度，必要时 Padding）-CUDA Graph Replay
        # Q： 为什么可以是向上而不能向下？ -- 显存地址、Tensor Shape 和计算操作之间的关系。
                    # static_x 地址：0x1000
                    # y 地址：       0x5000
                    # z 地址：       0x6000

                    # 计算操作：
                    # 0x1000 -> 矩阵乘法 -> 0x5000
                    # 0x5000 -> ReLU     -> 0x6000
        # Q：为什么只对batch_size进行pad，不对seq长度进行pad？ 
        # if batch_size > self.compilation_config.max_cudagraph_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_cudagraph_capture_size

        return self.compilation_config.bs_to_padded_graph_size[batch_size]
        # 将实际的 batch_size 向上取整（Padding）到预捕获的 CUDA Graph 尺寸。
        # 例如，如果 max_cudagraph_capture_size=8，
        # 且 bs_to_padded_graph_size={1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 7: 8, 8: 8}，那么：
        # pad_for_cudagraph(3) -> 4
        # CUDA Graph 只能在特定的 batch size 下运行。
        # 这个方法通过查表 (bs_to_padded_graph_size) 找到大于等于当前 batch size 的最小合法尺寸。
        # 如果请求的 batch size 超过了最大捕获尺寸，会抛出 IndexError。

    @property
    def needs_dp_coordinator(self) -> bool:
        # 数据并行协调器判断
        """
        Determine if the DPCoordinator process is needed.

        The DPCoordinator is needed in two cases:
        1. For MoE models with DP > 1: to handle wave coordination
           (even in external LB mode, since wave coordination runs in the coordinator)
        2. For non-MoE models in internal/hybrid LB mode: to collect and publish
           queue stats for load balancing across DP ranks

        Returns:
            True if DPCoordinator process is needed, False otherwise.
        """

        # For non-MoE models, only need coordinator in internal/hybrid LB mode
        # (for stats collection).
        return self.parallel_config.data_parallel_size > 1 and (
            self.model_config is None
            or self.model_config.is_moe
            or not self.parallel_config.data_parallel_external_lb
        ) 
    # MoE 模型 且 data_parallel_size > 1：需要协调器来处理“波次协调（wave coordination）”。
    # 非 MoE 模型 且使用内部/混合负载均衡模式：需要协调器来收集和发布队列统计信息，以进行跨 DP 秩的负载均衡。

    def enable_trace_function_call_for_thread(self) -> None:
        # 程级函数追踪
        """
        Set up function tracing for the current thread,
        if enabled via the `VLLM_TRACE_FUNCTION` environment variable.
        """
        if envs.VLLM_TRACE_FUNCTION: # 仅当环境变量 VLLM_TRACE_FUNCTION 被设置时生效。
            tmp_dir = tempfile.gettempdir()
            # add username to tmp_dir to avoid permission issues
            tmp_dir = os.path.join(tmp_dir, getpass.getuser())
            filename = (
                f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                f"_thread_{threading.get_ident()}_at_{datetime.now()}.log"
            ).replace(" ", "_")
            # 在临时目录下生成包含进程 ID、线程 ID 和时间戳的唯一日志文件路径
            log_path = os.path.join(
                tmp_dir,
                "vllm",
                f"vllm-instance-{self.instance_id}",
                filename,
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            enable_trace_function_call(log_path) # 调用底层的 enable_trace_function_call 记录执行轨迹
            # 用于极其底层的性能瓶颈排查

    @staticmethod
    def _get_quantization_config( # 获取并校验模型的量化配置。
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        """Get the quantization config."""
        from vllm.platforms import current_platform

        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import get_quant_config

            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}"
                )
            quant_config.maybe_update_config(model_config.model)
            return quant_config
        return None

    @staticmethod # 静态方法不需要实例化对象就可以采用
    def get_quantization_config( # 获取并校验模型的量化配置。
        model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(
            copy.deepcopy(model_config), load_config
        )

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: list[str] | None = None,
    ) -> "VllmConfig":  # 注入新的 HuggingFace 配置并返回一个全新的 VllmConfig 实例
        # 将 HuggingFace的配置参数注入到VllmConfig
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config
        model_config.model_arch_config = model_config.get_model_arch_config()

        return replace(self, model_config=model_config)

    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """Set config attribute to default if not already set by user.

        Args:
            config_obj: Configuration object to update.
            key: Attribute name.
            value: Default value (static or callable).
        """
        if getattr(config_obj, key) is None:
            # Some config values are known before initialization and are
            # hard coded.
            # Other values depend on the user given configuration, so they are
            # implemented with lambda functions and decided at run time.
            setattr(config_obj, key, value(self) if callable(value) else value)

    # VllmConfig 的初始化后处理逻辑（Post-Init Logic）。
    # 在 Python 的 dataclass 中，__post_init__ 方法会在对象初始化完成后自动执行。
    # 由于 vLLM 的配置项极其复杂且相互依赖，
    # 这段代码承担了“配置管家”的角色：它负责应用默认值、处理跨模块的配置联动、以及进行严格的兼容性校验。

    # VllmConfig是以配置嵌套配置，汇聚了很多应用场景不同的配置，下面对这些配置的参数做了校验
    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """Apply optimization level defaults using self as root.

        Recursively applies values from defaults into nested config objects.
        Only fields present in defaults are overwritten.

        If the user configuration does not specify a value for a default field
        and if the default field is still None after all user selections are
        applied, then default values will be applied to the field. User speciied
        fields will not be overridden by the default.

        Args:
            defaults: Dictionary of default values to apply.
        """
        # “”“ ”使用self作为根应用优化级别默认值。递归地将默认值应用到嵌套配置对象中。
        # 只有默认值中存在的字段才会被覆盖。如果用户配置没有为默认字段指定值，
        # 并且在应用了所有用户选择后，默认字段仍然为None，则默认值将应用于该字段。
        # 默认情况下，用户指定的字段不会被覆盖。
        # Args:defaults：要应用的默认值字典。“”"

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            """Recursively apply defaults to config_obj, using self as root."""
            for key, value in config_defaults.items():
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    def _post_init_kv_transfer_config(self) -> None:
        """Update KVTransferConfig based on top-level configs in VllmConfig.

        Right now, this function reads the offloading settings from
        CacheConfig and configures the KVTransferConfig accordingly.
        """
        # 基于VllmConfig中的顶层配置更新KVTransferConfig。
        # 现在，这个函数从mcacheconfig中读取卸载设置，并相应地配置KVTransferConfig。
        # KV offloading is only activated when kv_offloading_size is set.
        if (kv_offloading_size := self.cache_config.kv_offloading_size) is None:
            return

        kv_offloading_backend = self.cache_config.kv_offloading_backend

        # If no KVTransferConfig is provided, create a default one.
        if self.kv_transfer_config is None:
            self.kv_transfer_config = KVTransferConfig()
        num_kv_ranks = (
            self.parallel_config.tensor_parallel_size
            * self.parallel_config.pipeline_parallel_size
        )

        if kv_offloading_backend == "native":
            self.kv_transfer_config.kv_connector = "OffloadingConnector"
            self.kv_transfer_config.kv_connector_extra_config.update(
                {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
            )
        elif kv_offloading_backend == "lmcache":
            self.kv_transfer_config.kv_connector = "LMCacheConnectorV1"
            kv_gb_per_rank = kv_offloading_size / num_kv_ranks
            self.kv_transfer_config.kv_connector_extra_config = {
                "lmcache.local_cpu": True,
                "lmcache.max_local_cpu_size": kv_gb_per_rank,
            }

        # This is the same for all backends
        self.kv_transfer_config.kv_role = "kv_both"

    def __post_init__(self):
        """Verify configs are valid & consistent with each other."""

        # To give each torch profile run a unique instance name.
        #给每个profile配置文件运行一个唯一的实例名。
        self.instance_id = f"{time.time_ns()}"

        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(self.load_config)

            self.parallel_config.is_moe_model = self.model_config.is_moe

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_model_config(self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config
            )

        executor_backend = self.parallel_config.distributed_executor_backend
        executor_supports_async_sched = executor_backend in (
            "mp",
            "uni",
            "external_launcher",
        )

        if self.scheduler_config.async_scheduling:
            # Async scheduling explicitly enabled, hard fail any incompatibilities.
            if self.parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "Async scheduling is not yet compatible with "
                    "pipeline_parallel_size > 1."
                )
            # Currently, async scheduling only support eagle speculative
            # decoding.
            if self.speculative_config is not None:
                if self.speculative_config.method not in get_args(EagleModelTypes):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP kind of speculative decoding."
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            if not executor_supports_async_sched:
                raise ValueError(
                    "Currently, async scheduling only supports `mp`, `uni`, or "
                    "`external_launcher` distributed executor backend, but you chose "
                    f"`{executor_backend}`."
                )
        elif self.scheduler_config.async_scheduling is None:
            # Enable async scheduling unless there is an incompatible option.
            if self.parallel_config.pipeline_parallel_size > 1:
                logger.warning_once(
                    "Async scheduling is not yet supported with "
                    "pipeline_parallel_size > 1 and will be disabled.",
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.method not in get_args(EagleModelTypes)
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif (
                self.speculative_config is not None
                and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend (only `mp`, `uni`, and "
                    "`external_launcher` are supported).",
                    executor_backend,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            else:
                self.scheduler_config.async_scheduling = True

        logger.info_once(
            "Asynchronous scheduling is %s.",
            "enabled" if self.scheduler_config.async_scheduling else "disabled",
        )

        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            if self.scheduler_config.async_scheduling:
                logger.info_once(
                    "Disabling NCCL for DP synchronization "
                    "when using async scheduling.",
                    scope="local",
                )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        from vllm.platforms import current_platform

        if (
            self.model_config is not None
            and self.scheduler_config.enable_chunked_prefill
            and self.model_config.dtype == torch.float32
            and current_platform.get_device_capability() == (7, 5)
        ):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels."
            )

        if (
            self.optimization_level > OptimizationLevel.O0
            and self.model_config is not None
            and self.model_config.enforce_eager
        ):
            logger.warning("Enforce eager set, overriding optimization level to -O0")
            self.optimization_level = OptimizationLevel.O0

        if self.compilation_config.backend == "eager" or (
            self.compilation_config.mode is not None
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        def has_blocked_weights():
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                self.compilation_config.backend == "inductor"
                and self.compilation_config.mode != CompilationMode.NONE
            ):
                self.compilation_config.custom_ops.append("none")
            else:
                self.compilation_config.custom_ops.append("all")

        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        self._apply_optimization_level_defaults(default_config)

        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.info(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.fuse_gemm_comms:
            self.compilation_config.pass_config.enable_sp = True
        if self.compilation_config.pass_config.enable_sp:
            if "-rms_norm" in self.compilation_config.custom_ops:
                logger.warning(
                    "RMS norm force disabled, sequence parallelism might break"
                )
            else:
                self.compilation_config.custom_ops.append("+rms_norm")

        if current_platform.support_static_graph_mode():
            # if cudagraph_mode has full cudagraphs, we need to check support
            if model_config := self.model_config:
                if (
                    self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                    and model_config.pooler_config is not None
                ):
                    logger.warning_once(
                        "Pooling models do not support full cudagraphs. "
                        "Overriding cudagraph_mode to PIECEWISE."
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                elif (
                    model_config.is_encoder_decoder
                    and self.compilation_config.cudagraph_mode
                    not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
                ):
                    logger.info_once(
                        "Encoder-decoder models do not support %s. "
                        "Overriding cudagraph_mode to FULL_DECODE_ONLY.",
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = (
                        CUDAGraphMode.FULL_DECODE_ONLY
                    )

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # override related settings when enforce eager
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                self.compilation_config.cudagraph_num_of_warmups = 1

            self._set_cudagraph_sizes()
        else:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if self.cache_config.kv_sharing_fast_prefill:
            if (
                self.speculative_config is not None
                and self.speculative_config.use_eagle()
            ):
                raise ValueError(
                    "Fast prefill optimization for KV sharing is not "
                    "compatible with EAGLE as EAGLE requires correct logits "
                    "for all tokens while fast prefill gives incorrect logits "
                    "for prompt tokens."
                )

            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings."
            )
        # TODO: Move after https://github.com/vllm-project/vllm/pull/26847 lands
        self._set_compile_ranges()

        if (
            self.model_config
            and self.model_config.architecture == "WhisperForConditionalGeneration"
            and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            logger.warning(
                "Whisper is known to have issues with "
                "forked workers. If startup is hanging, "
                "try setting 'VLLM_WORKER_MULTIPROC_METHOD' "
                "to 'spawn'."
            )

        if (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
            and not self.cache_config.enable_prefix_caching
        ):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled. "
                "Use --enable-prefix-caching to enable."
            )
        if (
            self.kv_events_config is not None
            and self.kv_events_config.publisher != "null"
            and not self.kv_events_config.enable_kv_cache_events
        ):
            logger.warning(
                "KV cache events are disabled, "
                "but the scheduler is configured to publish them. "
                "Modify KVEventsConfig.enable_kv_cache_events "
                "to True to enable."
            )
        current_platform.check_and_update_config(self)

        # If DCP, ensure the block size is right.
        if self.parallel_config.decode_context_parallel_size > 1:
            if self.parallel_config.dcp_kv_cache_interleave_size > 1 and (
                self.parallel_config.cp_kv_cache_interleave_size
                != self.parallel_config.dcp_kv_cache_interleave_size
            ):
                self.parallel_config.cp_kv_cache_interleave_size = (
                    self.parallel_config.dcp_kv_cache_interleave_size
                )
                logger.warning_once(
                    "cp_kv_cache_interleave_size is overridden by dcp_kv_cache"
                    "_interleave_size. And dcp-kv-cache-interleave-size will be "
                    "deprecated when PCP is fully supported."
                )
            assert (
                self.parallel_config.cp_kv_cache_interleave_size
                <= self.cache_config.block_size
                and self.cache_config.block_size
                % self.parallel_config.cp_kv_cache_interleave_size
                == 0
            ), (
                f"Block_size({self.cache_config.block_size}) should be greater "
                "than or equal to and divisible by cp_kv_cache_interleave_size "
                f"({self.parallel_config.cp_kv_cache_interleave_size})."
            )

        # Do this after all the updates to compilation_config.mode
        effective_dp_size = (
            self.parallel_config.data_parallel_size
            if self.model_config is None or self.model_config.is_moe
            else 1
        )
        self.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=self.parallel_config.all2all_backend,
            data_parallel_size=effective_dp_size,
        )

        if self.compilation_config.pass_config.enable_sp:
            # With pipeline parallelism or dynamo partitioning,
            # native rms norm tracing errors due to incorrect residual shape.
            # Use custom rms norm to unblock. In the future,
            # the pass will operate on higher-level IR to avoid the issue.
            # TODO: https://github.com/vllm-project/vllm/issues/27894
            if self.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                logger.warning(
                    "Sequence parallelism is enabled, but running in wrong "
                    "vllm compile mode: %s.",
                    self.compilation_config.mode,
                )

            is_fullgraph = (
                self.compilation_config.use_inductor_graph_partition
                or len(self.compilation_config.splitting_ops) == 0
            )
            if self.parallel_config.pipeline_parallel_size > 1 or not is_fullgraph:
                if "-rms_norm" not in self.compilation_config.custom_ops:
                    self.compilation_config.custom_ops.append("+rms_norm")
                else:
                    regime = (
                        "Dynamo partition"
                        if not is_fullgraph
                        else "pipeline parallelism"
                    )
                    logger.warning_once(
                        "Sequence parallelism not supported with "
                        "native rms_norm when using %s, "
                        "this will likely lead to an error.",
                        regime,
                    )

        # final check of cudagraph mode after all possible updates
        if current_platform.is_cuda_alike():
            if (
                self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                and self.model_config is not None
                and not self.model_config.disable_cascade_attn
                and not self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()  # noqa: E501
            ):
                logger.warning_once(
                    "No piecewise cudagraph for executing cascade attention."
                    " Will fall back to eager execution if a batch runs "
                    "into cascade attentions."
                )

            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert self.compilation_config.mode == CompilationMode.VLLM_COMPILE, (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )

        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
            ], (
                "Microbatching currently only supports the deepep_low_latency and "
                f"deepep_high_throughput all2all backend. {a2a_backend} is not "
                "supported. To fix use --all2all-backend=deepep_low_latency or "
                "--all2all-backend=deepep_high_throughput and install the DeepEP"
                " kernels."
            )

            if not self.model_config.disable_cascade_attn:
                self.model_config.disable_cascade_attn = True
                logger.warning_once("Disabling cascade attention when DBO is enabled.")

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        # Hybrid KV cache manager (HMA) runtime rules:
        # - Explicit enable (--no-disable-kv-cache-manager): error if runtime
        #   disables it
        # - No preference: auto-disable for unsupported features (e.g. kv connector)
        # - Explicit disable (--disable-kv-cache-manager): always respect it
        need_disable_hybrid_kv_cache_manager = False
        # logger should only print warning message for hybrid models. As we
        # can't know whether the model is hybrid or not now, so we don't log
        # warning message here and will log it later.
        if not current_platform.support_hybrid_kv_cache():
            # Hybrid KV cache manager is not supported on non-GPU platforms.
            need_disable_hybrid_kv_cache_manager = True
        if self.kv_events_config is not None:
            # Hybrid KV cache manager is not compatible with KV events.
            need_disable_hybrid_kv_cache_manager = True
        if (
            self.model_config is not None
            and self.model_config.attention_chunk_size is not None
        ):
            if (
                self.speculative_config is not None
                and self.speculative_config.use_eagle()
            ):
                # Hybrid KV cache manager is not yet supported with chunked
                # local attention + eagle.
                need_disable_hybrid_kv_cache_manager = True
            elif not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
                logger.warning(
                    "There is a latency regression when using chunked local"
                    " attention with the hybrid KV cache manager. Disabling"
                    " it, by default. To enable it, set the environment "
                    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
                )
                # Hybrid KV cache manager is not yet supported with chunked
                # local attention.
                need_disable_hybrid_kv_cache_manager = True

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Default to disable HMA, but only if the user didn't express a preference.
            if self.kv_transfer_config is not None:
                # NOTE(Kuntai): turn HMA off for connector unless specifically enabled.
                need_disable_hybrid_kv_cache_manager = True
                logger.warning(
                    "Turning off hybrid kv cache manager because "
                    "`--kv-transfer-config` is set. This will reduce the "
                    "performance of vLLM on LLMs with sliding window attention "
                    "or Mamba attention. If you are a developer of kv connector"
                    ", please consider supporting hybrid kv cache manager for "
                    "your connector by making sure your connector is a subclass"
                    " of `SupportsHMA` defined in kv_connector/v1/base.py and"
                    " use --no-disable-hybrid-kv-cache-manager to start vLLM."
                )
            self.scheduler_config.disable_hybrid_kv_cache_manager = (
                need_disable_hybrid_kv_cache_manager
            )
        elif (
            self.scheduler_config.disable_hybrid_kv_cache_manager is False
            and need_disable_hybrid_kv_cache_manager
        ):
            raise ValueError(
                "Hybrid KV cache manager was explicitly enabled but is not "
                "supported in this configuration. Consider omitting the "
                "--no-disable-hybrid-kv-cache-manager flag to let vLLM decide"
                " automatically."
            )

        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # Default to enable HMA if not explicitly disabled by user or logic above.
            self.scheduler_config.disable_hybrid_kv_cache_manager = False

        if self.compilation_config.debug_dump_path:
            self.compilation_config.debug_dump_path = (
                self.compilation_config.debug_dump_path.absolute().expanduser()
            )
        if envs.VLLM_DEBUG_DUMP_PATH is not None:
            env_path = Path(envs.VLLM_DEBUG_DUMP_PATH).absolute().expanduser()
            if self.compilation_config.debug_dump_path:
                logger.warning(
                    "Config-specified debug dump path is overridden"
                    " by VLLM_DEBUG_DUMP_PATH to %s",
                    env_path,
                )
            self.compilation_config.debug_dump_path = env_path

        def has_blocked_weights():
            if self.quant_config is not None:
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # Enable quant_fp8 CUDA ops (TODO disable in follow up)
        # On H100 the CUDA kernel is faster than
        # native implementation
        # https://github.com/vllm-project/vllm/issues/25094
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # Handle the KV connector configs
        self._post_init_kv_transfer_config()

    def update_sizes_for_sequence_parallelism(self, possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled",
                removed_sizes,
                self.parallel_config.tensor_parallel_size,
            )

        return [
            size
            for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_cudagraph_sizes(self):
        """
        vLLM defines the default candidate list of batch sizes for CUDA graph
        capture as:

        ```python
        max_graph_size = min(max_num_seqs * 2, 512)
        # 1, 2, 4, then multiples of 8 up to 256 and then multiples of 16
        # up to max_graph_size
        cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_graph_size + 1, 16))

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in ascending order).

        These sizes are used to capture and reuse CUDA graphs for
        performance-critical paths (e.g., decoding). Capturing enables
        significantly faster kernel dispatch by avoiding Python overhead. The
        list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
        most GPUs), which controls the total allowed number of tokens in a
        batch. Since each sequence may have a variable number of tokens, the
        maximum usable batch size will depend on actual sequence lengths.

        Example:
            With `max_num_batched_tokens = 8192`, and typical sequences
            averaging ~32 tokens, most practical batch sizes fall below 256.
            However, the system will still allow capture sizes up to 512 if
            shape and memory permit.

        Note:
            If users explicitly specify cudagraph capture sizes in the
            compilation config, those will override this default logic.
            At runtime:

            - If batch size <= one of the `cudagraph_capture_sizes`, the closest
            padded CUDA graph will be used.
            - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
            not be used.
        """

        if (
            self.model_config is not None
            and not self.model_config.enforce_eager
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # determine the initial max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            if max_cudagraph_capture_size is None:
                decode_query_len = 1
                if (
                    self.speculative_config
                    and self.speculative_config.num_speculative_tokens
                ):
                    decode_query_len += self.speculative_config.num_speculative_tokens
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
                )
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )

            # determine the cudagraph_capture_sizes
            if self.compilation_config.cudagraph_capture_sizes is not None:
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                # de-duplicate the sizes provided by the config
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                # sort to make sure the sizes are in ascending order
                cudagraph_capture_sizes.sort()
            else:
                cudagraph_capture_sizes = [
                    i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                ]
                if max_cudagraph_capture_size >= 8:
                    # Step size 8 for small batch sizes, up to 256(not included)
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # Step size 16 for larger batch sizes
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )

            if (
                self.parallel_config.tensor_parallel_size > 1
                and self.compilation_config.pass_config.enable_sp
            ):
                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                    cudagraph_capture_sizes
                )

            # user-specific compilation_config.max_cudagraph_capture_size get
            # truncated to valid_max_size when they are inconsistent.
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            if (
                self.compilation_config.max_cudagraph_capture_size is not None
                and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # raise error only when both two flags are user-specified
                # and they are inconsistent with each other
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )

                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d",
                    valid_max_size,
                )
            # always set the final max_cudagraph_capture_size
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                # If users have specified capture sizes, we only need to
                # compare the lens before and after modification since the modified
                # list is only the subset of the original list.
                logger.warning(
                    (
                        "cudagraph_capture_sizes specified in compilation_config"
                        " %s is overridden by config %s"
                    ),
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            # always write back the final sizes
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

        else:
            # no cudagraph in use
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # complete the remaining process.
        self.compilation_config.post_init_cudagraph_sizes()

    def _set_compile_ranges(self):
        """
        Set the compile ranges for the compilation config.
        """
        compilation_config = self.compilation_config
        computed_compile_ranges_split_points = []

        # The upper bound of the compile ranges is the max_num_batched_tokens
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        if max_num_batched_tokens is not None:
            computed_compile_ranges_split_points.append(max_num_batched_tokens)

        # Add the compile ranges for flashinfer
        if compilation_config.pass_config.fuse_allreduce_rms:
            tp_size = self.parallel_config.tensor_parallel_size
            max_size = compilation_config.pass_config.flashinfer_max_size(tp_size)
            if max_size is not None:
                max_token_num = max_size // (
                    self.model_config.get_hidden_size()
                    * self.model_config.dtype.itemsize
                )
                if (
                    max_num_batched_tokens is not None
                    and max_token_num < max_num_batched_tokens
                ):
                    computed_compile_ranges_split_points.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below allreduce-rms fusion threshold, "
                        "allreduce-rms fusion will be enabled for all num_tokens."
                    )

        if compilation_config.compile_ranges_split_points is not None:
            for x in compilation_config.compile_ranges_split_points:
                assert isinstance(x, int)
                assert x > 0, f"Invalid compile range split point: {x}"
                if (
                    max_num_batched_tokens is not None
                    and x < max_num_batched_tokens
                    and x > 1
                ):
                    computed_compile_ranges_split_points.append(x)
        compilation_config.compile_ranges_split_points = sorted(
            computed_compile_ranges_split_points
        )

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import SequenceClassificationConfig

            SequenceClassificationConfig.verify_and_update_config(self)

        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
            self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                "runai_streamer",
                "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from S3, 'load_format' "
                    f"must be 'runai_streamer' or 'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

    def compile_debug_dump_path(self) -> Path | None:
        """Returns a rank-aware path for dumping
        torch.compile debug information.
        """
        if self.compilation_config.debug_dump_path is None:
            return None
        tp_rank = self.parallel_config.rank
        dp_rank = self.parallel_config.data_parallel_index
        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
        path = self.compilation_config.debug_dump_path / append_path
        return path

    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"enable_return_routed_experts={self.model_config.enable_return_routed_experts}, "  # noqa
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"structured_outputs_config={self.structured_outputs_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"enable_chunked_prefill={self.scheduler_config.enable_chunked_prefill}, "  # noqa
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}"
        )

    @model_validator(mode="after")
    def validate_mamba_block_size(self) -> "VllmConfig":
        if self.model_config is None:
            return self
        mamba_block_size_is_set = (
            self.cache_config.mamba_block_size is not None
            and self.cache_config.mamba_block_size != self.model_config.max_model_len
        )
        if mamba_block_size_is_set and not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "--mamba-block-size can only be set with --enable-prefix-caching"
            )
        return self


_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from vllm.compilation.counter import compilation_counter

    num_models_seen = compilation_counter.num_models_seen
    try:
        # Clear the compilation config cache when context changes.
        # This is needed since the old config may have been accessed
        # and cached before the new config is set.
        get_cached_compilation_config.cache_clear()

        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            vllm_config.compilation_config.custom_op_log_check()

        if (
            check_compile
            and vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and compilation_counter.num_models_seen == num_models_seen
        ):
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model,
            )
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    return get_current_vllm_config().compilation_config


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_vllm_config' pytest fixture from tests/conftest.py."
        )
    return _current_vllm_config


def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config


T = TypeVar("T")


def get_layers_from_vllm_config(
    vllm_config: VllmConfig,
    layer_type: type[T],
    layer_names: list[str] | None = None,
) -> dict[str, T]:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vLLM config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    if layer_names is None:
        layer_names = list(vllm_config.compilation_config.static_forward_context.keys())

    forward_context = vllm_config.compilation_config.static_forward_context

    return {
        layer_name: forward_context[layer_name]
        for layer_name in layer_names
        if isinstance(forward_context[layer_name], layer_type)
    }
