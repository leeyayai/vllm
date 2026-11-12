# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

from vllm.v1.request import Request, RequestStatus


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.
    从列表中移除所有属于 items_to_remove 集合的元素。

    This method optimizes for the common case of removing a single item,
    此方法针对最常见的单元素移除场景进行了优化，
    falling back to list comprehension for multiple items.
    多元素移除时则回退到列表推导式。

    Args:
        lst: The list to remove items from
        lst：要从中移除元素的列表。
        items_to_remove: Set of items to remove
        items_to_remove：要移除的元素集合。

    Returns:
        Either the modified original list (for single item removal) or
        单个元素时返回修改后的原列表；多个元素时返回
        a new list (for multiple item removal). Callers should use the
        一个新列表。调用方应使用
        returned value.
        返回值。

    Note:
        For single item removal, this modifies the original list in-place
        注意：移除单个元素时会原地修改原列表
        and returns it. For multiple items, it creates and returns a new list.
        并返回该列表；移除多个元素时会创建并返回新列表。
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        # 单元素移除的快速路径（最常见的情况）。
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    # 移除多个元素时使用列表推导式。
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
