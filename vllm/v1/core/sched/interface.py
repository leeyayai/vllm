# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class SchedulerInterface(ABC):  # 调度器接口
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.
        调度本步骤需要处理的请求。

        The scheduling decision is made at the iteration level. Each scheduling
        调度决策以迭代为单位。每个调度步骤对应模型的一次前向计算，
        step corresponds to a single forward pass of the model. Therefore, this
        因此引擎的主循环会反复调用
        method is called repeatedly by a busy loop in the engine.
        此方法。

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        Scheduler 实际上会生成一个 `{req_id: num_tokens}` 字典，
        that specifies how many tokens to process for each request in this
        指明本调度步骤中每个请求要处理的 token 数量。
        scheduling step. For example, num_tokens can be as large as the number
        例如，新请求的 num_tokens 可以等于 prompt token 数，
        of prompt tokens for new requests, or it can be 1 for the requests that
        自回归逐 token 生成的请求则可以为 1；在
        are auto-regressively generating new tokens one by one. Otherwise, it
        分块预填充、前缀缓存、推测解码等场景下，
        can be somewhere in between in case of chunked prefills, prefix caching,
        该值也可以介于两者之间。
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        此外，Scheduler 还会返回每个请求或整个 batch 的
        or the batch as a whole. The model runner will use this information in
        有用信息，Model Runner 会在准备模型输入时使用这些信息。
        preparing inputs to the model.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            返回包含已调度请求相关信息的 SchedulerOutput 对象。
            requests.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.
        根据 Model Runner 的输出更新 Scheduler 状态。

        This method is called after the model runner has processed the scheduled
        Model Runner 处理完已调度请求后会调用此方法。输出包含生成的 token ID、
        requests. The model runner output includes generated token ids, draft
        下一步的草稿 token ID 等信息。Scheduler 使用这些信息更新自身状态，
        token ids for next step, etc. The scheduler uses this information to
        检查已结束的请求，
        update its states, checks the finished requests, and returns the output
        并返回
        for each request.
        每个请求的输出。

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            返回客户端索引到 EngineCoreOutputs 对象的映射，其中包含
            outputs for each request originating from that client.
            来自该客户端的各请求输出。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
        """Update requests with newly generated draft token ids, applying
        使用新生成的草稿 token ID 更新请求，并在需要时应用
        structured output grammar validation if needed.
        结构化输出的语法校验。

        Args:
            draft_token_ids: The input draft token ids for each request.
            draft_token_ids：每个请求对应的输入草稿 token ID。
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update scheduler output with newly generated draft token ids, applying
        使用新生成的草稿 token ID 更新 Scheduler 输出，并在需要时应用
        structured output grammar validation if needed.
        结构化输出的语法校验。

        Args:
            draft_token_ids: The input draft token ids for each request.
            draft_token_ids：每个请求对应的输入草稿 token ID。
            scheduler_output: Update the given scheduler_output
            scheduler_output：要更新的 scheduler_output，
                with the corresponding draft token ids.
                写入相应的草稿 token ID。
        """
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.
        将新请求加入 Scheduler 的内部队列。

        Args:
            request: The new request being added.
            request：要添加的新请求。
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "RequestStatus",
    ) -> None:
        """Finish the requests in the scheduler's internal queue. If the request
        结束 Scheduler 内部队列中的请求。如果请求
        is not in the queue, this method will do nothing.
        不在队列中，此方法不执行任何操作。

        This method is called in two cases:
        以下两种情况下会调用此方法：
        1. When the request is aborted by the client.
        1. 客户端中止请求时。
        2. When the frontend process detects a stop string of the request after
        2. 前端进程对生成的 token 解码后检测到请求的停止字符串时。
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs.
            request_ids：单个请求 ID 或请求 ID 列表。
            finished_status: The finished status of the given requests.
            finished_status：这些请求结束后的状态。
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue.
        Scheduler 内部队列中未完成的请求数量。
        """
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        若 Scheduler 内部队列中存在未完成请求，则返回 True。
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        若存在需要清理的已完成请求，则返回 True。
        NOTE: This is different from `not self.has_unfinished_requests()`.
        注意：这与 `not self.has_unfinished_requests()` 含义不同。

        The scheduler maintains an internal list of the requests finished in the
        Scheduler 会维护一个内部列表，记录上一步中已结束的请求。
        previous step. This list is returned from the next call to schedule(),
        下一次调用 schedule() 时会返回此列表，
        to be sent to the model runner in the next step to clear cached states
        并在下一步发送给 Model Runner，以清理这些请求的缓存状态。
        for these finished requests.

        This method checks if this internal list of finished requests is
        此方法检查该已完成请求列表是否
        non-empty. This information is useful for DP attention.
        非空。该信息可用于 DP attention。
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        若存在未完成请求，或尚未通过 SchedulerOutputs 返回的已完成请求，
        not yet returned in SchedulerOutputs."""
        #则返回 True。
        return self.has_unfinished_requests() or self.has_finished_requests()

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache for KV cache.
        重置 KV cache 的前缀缓存。

        This is particularly required when the model weights are live-updated.
        模型权重在线更新时尤其需要执行此操作。

        Args:
            reset_running_requests: If True, all the running requests will be
            reset_running_requests：为 True 时，将所有运行中的请求
                preempted and moved to the waiting queue. Otherwise, this method
                抢占并移入等待队列；否则仅在没有运行请求占用 KV cache 时
                will only reset the KV prefix cache when there is no running request
                才重置 KV 前缀缓存。
                taking KV cache.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs).
        返回 (num_running_reqs, num_waiting_reqs)，即运行中和等待中的请求数。
        """
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> Optional["SchedulerStats"]:
        """Make a SchedulerStats object for logging.
        创建用于日志记录的 SchedulerStats 对象。

        The SchedulerStats object is created for every scheduling step.
        每个调度步骤都会创建一个 SchedulerStats 对象。
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler.
        关闭 Scheduler。
        """
        raise NotImplementedError

    def get_kv_connector(self) -> Optional["KVConnectorBase_V1"]:
        return None
