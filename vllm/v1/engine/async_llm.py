# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any, cast

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class AsyncLLM(EngineClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
        核心配置参数
            vllm_config: global configuration.
            全局配置对象。它包含了模型路径、张量并行度、量化方式等所有关于模型和运行环境的核心设置。
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            执行器类（Executor 的实现）。决定了模型如何被加载和计算，
            例如 MultiprocExecutor（多进程执行器）或 GpuExecutor。
            mm_registry: Multi-modal registry.
            多模态注册表（Multi-modal registry）。
            如果模型支持多模态（如图像+文本），这个注册表用于管理多模态数据的处理逻辑。
        上下文参数
            usage_context: Usage context of the LLM.
            使用上下文。用于标识当前 LLM 的使用场景（例如：离线批量推理、在线 API 服务等），
            引擎可能会根据此上下文采用不同的调度策略或资源分配方式。
            use_cached_outputs: Whether to use cached outputs.
            否使用缓存的输出结果;开启后，引擎会缓存之前的生成结果
            （通常用于 Speculative Decoding 投机采样或特定的 Prefix Caching 场景），
            避免重复计算，从而提升推理速度。
        日志与监控参数   
            log_stats: Whether to log stats.
             是否记录引擎的运行统计信息（如吞吐量、延迟、队列长度等）。
            log_requests: Whether to log requests.
            是否记录请求日志。
            开启后，引擎会在控制台或日志文件中打印每个进入系统的请求详情（如 prompt 内容、请求 ID 等）。
            这在调试阶段非常有用，但在生产环境中通常建议关闭，以避免日志量过大影响性能。
            start_engine_loop: Whether to start the engine loop.
            是否启动引擎的主循环。
            控制引擎在初始化后是否立即开始处理请求的后台循环。通常设为 True；
            但在某些高级用法（如单元测试、手动控制引擎生命周期、或作为子组件嵌入其他系统）时，
            可能会设为 False，由外部代码手动触发循环。
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.
            引擎的自定义统计日志记录器。
            允许用户传入自定义的类来收集引擎的运行指标（如吞吐量、延迟、队列长度等）。
            如果不提供，引擎会使用默认的统计记录器。
            (vLLM V1 架构中，统计记录器的基类接口还不稳定)

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_tokenizer_from_config(self.model_config)

        self.input_processor = InputProcessor(self.vllm_config, tokenizer) # self.vllm_config参数校验
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        ) # 这都是在初始化，并非运行的流程

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        # Pause / resume state for async RL workflows.
        self._pause_cond = asyncio.Condition()
        self._paused = False

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()  # 启动输出处理任务
        except RuntimeError:
            pass

        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):
            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    #提供了两个 @classmethod（类方法）作为创建引擎实例的入口，它们最终都会调用底层的 cls(...) 来初始化对象。
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":  
        # Create the LLMEngine. # 初始化 LLMEngine 引擎，加载模型权重、初始化 KV Cache 显存块、创建调度器（Scheduler）和模型执行器（ModelExecutor）。
        # 当用户已经构建好了完整的 VllmConfig 对象，或者需要自定义客户端参数（如分布式场景下的 client_addresses, client_count）时，直接使用此方法。
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # 用于 API Server 启动时，接收命令行解析出的参数（AsyncEngineArgs）

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM. # 初始化 AsyncLLM 引擎，加载模型权重、初始化 KV Cache 显存块、创建调度器（Scheduler）和模型执行器（ModelExecutor）。
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        shutdown_prometheus() # 关闭 Prometheus 服务器，释放资源。

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self.engine_core.get_supported_tasks_async()

    async def add_request(  # 添加请求，进入引擎的等待队列， 在encode和generate方法里调用
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and not is_pooling
            and params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        _validate_truncation_size(
            self.model_config.max_model_len,
            params.truncate_prompt_tokens,
            tokenization_kwargs,
        )

        # Convert Input --> Request. # 将输入转换为请求对象，用于引擎处理。
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            if prompt_text is not None:
                raise ValueError(
                    "should only provide prompt_text with EngineCoreRequest"
                )
            request = self.input_processor.process_inputs( # 将请求转为tokenids列表
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        self.input_processor.assign_request_id(request)# 给当前请求分配一个唯一的 request_id
        #      request = EngineCoreRequest(  # 举例
        #     request_id = None,  -- 填充这个
        #     prompt_token_ids = [101, 234, 567]
        # )

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # 启动一个异步的 Output Handler，负责从 EngineCore 接收推理结果，并将结果分发给对应的请求
        # 因为 vLLM 的请求处理是异步并发的
        # 情况
        # Request A ─┐
        # Request B ─┼──→ EngineCore
        # Request C ─┘
        #                 |
        #                 | 推理结果
        #                 v
        #             OutputHandler
        #                 |
        #         ┌────────┼────────┐
        #         v        v        v
        #     Queue A  Queue B  Queue C
        # EngineCore 产生的输出需要被正确分发到对应请求的输出队列。
        # 不在 AsyncLLM.__init__() 中启动 OutputHandler，而是在第一次调用 add_request() 时启动。
        self._run_output_handler()


        # Respect pause state before accepting new requests.
        # 检查 vLLM 当前是否处于暂停状态。
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        # Create a new output collector for the request.
        # 给当前请求创建一个专属的输出收集器。
        # 因为一个请求的生成过程可能会产生很多次输出。
        # Request A
        #     |
        #     | 第一次生成
        #     v
        #     Token 1
        #     |
        #     | 第二次生成
        #     v
        #     Token 2
        #     |
        #     | 第三次生成
        #     v
        #     Token 3
        # params.output_kind：决定输出形式，例如累积输出还是增量输出等。
        # request.request_id：标识这个收集器属于哪个请求。
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue) # 添加请求
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

    #     request = EngineCoreRequest(  n=3 对于一个问题用户希望有3个回答
    #     request_id="parent-001",
    #     prompt_token_ids=[101, 102, 103],
    #     sampling_params=SamplingParams(n=3)
    # )
        # Fan out child requests (for n>1).
        # Child 0:
        #     request_id = child-0
        #     prompt_token_ids = [101, 102, 103]

        # Child 1:
        #     request_id = child-1
        #     prompt_token_ids = [101, 102, 103]

        # Child 2:
        #     request_id = child-2
        #     prompt_token_ids = [101, 102, 103]
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id # 建立子请求
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            ) # 添加子请求
        return queue

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process). 在当前 Driver Process 的 OutputProcessor 中登记请求。
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # 为什么要先添加到 OutputProcessor？
        # 因为请求发送给 EngineCore 后，EngineCore 可能很快就产生输出。
        # Driver Process 必须提前登记好这个请求，才能知道后面收到的输出应该交给哪个请求对应的 queue。
        # 可以理解为：先建立输出接收档案，再把请求发送出去。

        # Add the EngineCoreRequest to EngineCore (separate process). 把 EngineCoreRequest 发送给独立的 EngineCore 进程。
        # EngineCore 会根据请求的 sampling_params 进行推理。进入 Scheduler 的 waiting 队列,再进入running队列
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(  # 主函数，生成请求
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """
        # generate() 本身不负责模型推理，也不负责 Scheduler 调度。它是一个异步生成器，主要负责提交请求，
        # 然后持续从输出队列中取出结果，逐步返回给调用者

        q: RequestOutputCollector | None = None # 初始化一个空的 RequestOutputCollector 实例
        try:
            q = await self.add_request(  # 添加请求
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
            )  # 等待 add_request 里的返回的 RequestOutputCollector

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:  # 循环从输出队列中取出结果，直到请求完成
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                assert isinstance(out, RequestOutput)
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        + "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""
        # 后台异步输出处理循环（Background Output Handler），负责不断从 EngineCore 获取模型推理结果，
        # 经过 OutputProcessor 处理后，把结果放入之前创建的 RequestOutputCollector 队列中。

        # generate()：消费者，等待并获取结果。
        # OutputHandler：后台搬运工，持续接收和处理结果。
        # OutputProcessor：负责处理结果、匹配请求、组织输出。
        # RequestOutputCollector：保存每个请求的输出，供 generate() 获取。
        # EngineCore：真正负责调度和执行模型推理。

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager
        input_processor = self.input_processor
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        async def output_handler():
                #              AsyncLLM
                #                 |
                #     一个 OutputHandler
                #                 |
                #         EngineCore
                #                 |
                #     ┌──────────┼──────────┐
                #     v          v          v
                # Req A      Req B      Req C
                #     |          |          |
                #     v          v          v
                # Queue A    Queue B    Queue C
    # EngineCore 可以一次返回多个请求的输出，OutputHandler 可以统一处理这些输出，再按照 request_id 分发给各自的请求队列。
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=input_processor.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """
        Pause generation to allow model weight updates.

        New generation/encoding requests are blocked until resume.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight
                requests to finish before pausing. When ``False`` (default),
                immediately aborts any in-flight requests.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
                Default is ``True`` (clear caches).
        """

        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if not wait_for_inflight_requests:
            request_ids = list(self.output_processor.request_states.keys())
            if request_ids:
                await self.abort(request_ids, internal=True)

        # Wait for running requests to drain before clearing cache.
        if self.output_processor.has_unfinished_requests():
            await self.output_processor.wait_for_requests_to_drain()

        # Clear cache
        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""

        async with self._pause_cond:
            return self._paused

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:

        # generate()：文本生成任务（Generation），例如输入一段话，让模型继续生成文本。
        # encode()：Pooling 任务，例如 Embedding、文本向量提取、分类等。
        # 文本生成：
        # async def generate(...):
        #     q = await self.add_request(
        #         request_id,
        #         prompt,
        #         sampling_params,
        #         ...
        #     )    
        #   向量提取
        #  async def encode(...):
        #     q = await self.add_request(
        #         request_id,
        #         prompt,
        #         pooling_params,
        #         ...
        #     ) 
        
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.

        NOTE: truncate_prompt_tokens is deprecated in v0.14.
        TODO: Remove truncate_prompt_tokens in v0.15.
        """

        q: RequestOutputCollector | None = None
        try:
            if truncate_prompt_tokens is not None:
                warnings.warn(
                    "The `truncate_prompt_tokens` parameter in `AsyncLLM.encode()` "
                    "is deprecated and will be removed in v0.15. "
                    "Please use `pooling_params.truncate_prompt_tokens` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_processor.tokenizer

    async def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None  # type: ignore

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        coros = [self.engine_core.profile_async(True)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def sleep(self, level: int = 1) -> None:
        await self.reset_prefix_cache()
        await self.engine_core.sleep_async(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return
        logger.info(
            "Waiting for requests to drain before scaling up to %s engines...",
            new_data_parallel_size,
        )
        await self.wait_for_requests_to_drain(drain_timeout)
        logger.info(
            "Requests have been drained, proceeding with scale to %s engines",
            new_data_parallel_size,
        )
        await self.engine_core.scale_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size

        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()
