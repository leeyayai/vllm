# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@bc_linter_include
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@bc_linter_include
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # 对于不在 resumed_req_ids 中的请求 ID，new_block_ids 会追加到
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # 该请求现有的 block ID 列表末尾；对于集合中的请求，new_block_ids 会作为
    # request's block IDs instead of appending to the existing block IDs.
    # 请求的完整 block ID 列表使用，而不是追加到现有列表。
    resumed_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # 注意（woosuk）：new_token_ids 仅用于流水线并行。
    # When PP is not used, new_token_ids will be empty.
    # 未使用流水线并行（PP）时，new_token_ids 为空。
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # 对于上一步未调度的请求，将 token ID 传递给
    # connector. Won't contain requests that were scheduled in the prior step.
    # connector。此处不包含上一步已调度的请求。
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    # Version of dataclass repr with token IDs obfuscated.
    # 对 dataclass repr 的脱敏版本，其中 token ID 会被隐藏。
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """Cache mapping of req_id to num_output_tokens for O(1) lookup.

        This cached property is safe because CachedRequestData instances
        are created fresh each scheduling iteration and not mutated during
        computation of iteration details.
        """
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@bc_linter_include
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # 本轮首次被调度的请求列表。
    # We cache the request's data in each worker process, so that we don't
    # 请求数据会缓存在各 Worker 进程中，
    # need to re-send it every scheduling step.
    # 因此后续每一步无需重复发送完整数据。
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # 之前已经调度过的请求列表。
    # Since the request's data is already cached in the worker processes,
    # 由于请求数据已缓存在 Worker 进程中，
    # we only send the diff to minimize the communication cost.
    # 这里只发送增量信息以减少通信开销。
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # 请求 ID 到本轮调度 token 数的映射。
    # Number of tokens scheduled for each request.
    # 每个请求本轮被调度处理的 token 数。
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # 所有请求本轮调度 token 数的总和。
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # 请求 ID 到推测解码 token ID 列表的映射。
    # If a request does not have any spec decode tokens, it will not be
    # 如果请求没有推测解码 token，
    # included in the dictionary.
    # 则不会出现在此字典中。
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # 请求 ID 到本轮需要处理的 encoder 输入索引列表的映射。
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # 例如，若请求对应 [0, 1]，表示视觉 encoder 需要处理
    # to process that the request's 0-th and 1-th images in the current step.
    # 本轮请求中的第 0 张和第 1 张图像。
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # 每个 KV cache group 中所有请求共享的前缀 block 数量。
    # This can be used for cascade attention.
    # 该信息可用于级联注意力（cascade attention）。
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # 在上一步与当前步骤之间结束的请求 ID。
    # steps. This is used to notify the workers about the finished requests
    # 用于通知 Worker 哪些请求已结束，
    # so that they can free the cached states for those requests.
    # 以便 Worker 释放这些请求的缓存状态。
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # 需要从 encoder cache 中释放的 encoder 输出对应的 mm_hash 列表。
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request IDs that are preempted in this step.
    # 本轮被抢占的请求 ID 集合。
    # Only used for v2 model runner.
    # 仅供 v2 model runner 使用。
    preempted_req_ids: set[str] | None = None

    # Whether any of the scheduled requests use structured output.
    # 本轮调度的请求中是否包含结构化输出请求。
    # Set only in async scheduling case.
    # 仅在异步调度模式下设置。
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # 调度中的请求是否已拥有计算语法 bitmask 所需的全部输出 token。
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # Used for adjusting acceptance rate calculation.
    # 用于修正接受率计算。
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV Cache Connector metadata.
    # KV Cache Connector 的元数据。
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    # EC Cache Connector 的元数据。
    ec_connector_metadata: ECConnectorMetadata | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    # 结构化输出请求的 ID。
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    # bitmask 中各行的顺序与 structured_output_request_ids 一致。
    grammar_bitmask: "npt.NDArray[np.int32]"
