import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

# PR：https://github.com/vllm-project/vllm/pull/63458
class BudgetConfig(BaseModel):
    # 继承 BaseModel，主要是为了使用 Pydantic 提供的数据校验、类型检查、默认值管理和配置对象功能。
    
    model_config = ConfigDict( # 配置对象，用于指定模型的校验规则。
    extra="forbid", # 禁止出现未定义的配置项
    strict=True, # 严格类型检查
    frozen=True, # 配置对象创建后不可修改
    )

    high_budget: int = Field(default=8192, ge=1, 
                             description="每轮最大调度数，大于等于1")
    low_budget: int = Field(default=1024, ge=1, 
                            description="prefill和decode混合每轮最大调度数，大于等于1")
    enter_prefills:int=Field(default=2,ge=1,
                             description="有decode，且可运行prefill达到2个时，进入限流状态")
    exit_prefills:int=Field(default=0,ge=0,
                            description="限流状态下，prefill数量降到0时退出")
    min_dwell_steps:int=Field(default=4,ge=0,
                              description="两次压力状态切换至少间隔4个调度step")
    guard_enabled:bool=Field(default=True,
                             description="是否启用prefill进度保障")
    max_stall_ms:float=Field(default=250.0,ge=0.0,allow_inf_nan=False,
                             description="prefill超过多久没有进展认为它停滞，单位毫秒")
    guard_budget:int=Field(default=4096,ge=1,
                             description="触发保障时，临时提高本轮预算")
    guard_interval_steps:int=Field(default=8,ge=1,
                             description="两次触发保障至少间隔8个调度step")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BudgetConfig":
        if not self.low_budget <= self.guard_budget <= self.high_budget:
            raise ValueError("Require low_budget <= guard_budget <= high_budget")
        if self.exit_prefills >= self.enter_prefills:
            raise ValueError("Require exit_prefills < enter_prefills")
        return self


# -------------------如何决定调度预算------------------ 
class BudgetController:

    def __init__(self, config: BudgetConfig):
        self.config = config # 保存预算配置
        self.limited = False # 当前是否处于低预算限制模式--初始状态不限制 Prefill 预算
        self.step = 0 # 已经执行的决策次数
        self.last_switch = -config.min_dwell_steps # 上一次模式切换发生在哪一轮：低预算和高预算两种self.limited
        # 了让控制器在启动后第一轮就可以切换状态，不需要先等待一个完整的 dwell 周期。
        self.last_guard = -config.guard_interval_steps # 上一次触发保障发生在哪一轮
        # 了让控制器在启动后第一轮就可以触发保障，不需要先等待一个完整的 guard 周期。

    def decide(self, prefills: int, decodes: int, stall_ms: float) -> int:
        # 这段代码实际上有三个决策阶段：
        #     1. 判断是否进入或退出限制模式。
        #     2. 判断是否需要触发保护预算。
        #     3. 返回最终预算。
        cfg = self.config
        self.step += 1
        previous = self.limited
        # 这个策略的目标不是单纯限制 Prefill，而是协调 Prefill 和 Decode 的资源竞争
        if not prefills or not decodes:  # 没有prefill或decode，为High Budget模式
            self.limited = False
        elif self.step - self.last_switch >= cfg.min_dwell_steps:
            if not self.limited and prefills >= cfg.enter_prefills:
                self.limited = True # 进入低预算限制模式
            elif self.limited and prefills <= cfg.exit_prefills:
                self.limited = False # 退出低预算限制模式
        if previous != self.limited:
            self.last_switch = self.step # 记录模式切换时间
        # ----------------第三种预算：Guard Budget（保护预算）------------------ 
        if not self.limited:
            return cfg.high_budget # 非低预算限制模式，返回高预算
        # 如果进入了限制模式，继续判断：
        if (
            cfg.guard_enabled # 是否开启保护机制
            and stall_ms >= cfg.max_stall_ms  # 停滞了很久 
            and self.step - self.last_guard >= cfg.guard_interval_steps # 距离上次保护动作是否达到间隔
        ):
            self.last_guard = self.step
            return cfg.guard_budget  # 触发保障时，临时提高本轮预算
        return cfg.low_budget


class  AdaptiveBudgetScheduler(Scheduler):
    
    def __init__(self, vllm_config: VllmConfig, *args, **kwargs)->None:
        sc = vllm_config.scheduler_config
        if sc.async_scheduling or not sc.enable_chunked_prefill:
            raise ValueError("Require async_scheduling=False and enable_chunked_prefill=True")
        if(
            sc.is_multimodal_model
            or sc.runner_type!="generate"
            or vllm_config.model_config.is_encoder_decoder
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or vllm_config.kv_transfer_config is not None
            or vllm_config.ec_transfer_config is not None
            or vllm_config.parallel_config.world_size!=1
            or vllm_config.parallel_config.data_parallel_size!=1
        ):
            raise ValueError("This example requires single-GPU text generation")
        # 自定义调度器需要一些 vLLM 原生配置中没有的参数，从 additional_config 中获取
        # 这里要求它必须是 Python 字典。
        # additional_config = {
        # "adaptive_budget": {
        #     "high_budget": 8192,
        #     "low_budget": 1024,
        #     "enter_prefills": 2,
        #     "exit_prefills": 0,
        #     "min_dwell_steps": 4
        #     }
        # }
        additional=vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("additional_config must be a dictionary")
        self.budget_config = BudgetConfig.model_validate(
            additional.get("adaptive_budget", {})
        )
        # 检查预算是否超过调度器容量
        capacity = sc.max_num_scheduled_tokens or sc.max_num_batched_tokens
        if self.budget_config.high_budget > capacity:
            raise ValueError("high_budget exceeds the scheduler's configured capacity")
        super().__init__(vllm_config, *args, **kwargs)
        self.budget_controller = BudgetController(self.budget_config)
        self.prefill_progress: dict[str, tuple[Request, int, float]] = {}

        # 观察当前所有请求的执行状态，统计正在进行 Prefill 的请求数、Decode 请求数，并计算 Prefill 请求中最长的停滞时间
    def _observe(self, now: float) -> tuple[int, int, float]:
    # 返回 (prefills, decodes, oldest_stall_ms)
        prefills = decodes = 0 # 当前正在进行 Prefill 的请求数量,当前处于 Decode 阶段的 Running 请求数量
        oldest_stall_ms = 0.0 # 	Prefill 请求中最长的无进展时间，单位毫秒
        live: set[str] = set() # 用于记录当前仍然处于 Prefill 阶段的请求 ID。
        # 后面需要清理已经不再进行 Prefill 的请求的历史进度记录。
        for req_id, req in self.requests.items():
            if req.status not in (
                RequestStatus.WAITING,
                RequestStatus.RUNNING,
                RequestStatus.PREEMPTED,
            ):
                continue
            # ------ 如何判断请求处于 Prefill 还是 Decode------
            # req.num_prompt_tokens	原始 Prompt 的 Token 数量
            # req.num_tokens	当前请求累计的 Token 数量，包含 Prompt 和已经生成的 Token
            prefill_end = max(req.num_prompt_tokens, req.num_tokens - 1) # 包括抢占恢复的情况
            if req.num_computed_tokens < prefill_end:
                prefills += 1
                live.add(req_id)
                    #live = {
                        #     "request_001",
                        #     "request_002",
                        # }
                old_req, old_count, last_progress = self.prefill_progress.get(
                    req_id, (req, req.num_computed_tokens, now)
                ) # 获取请求之前的进度记录
                    # self.prefill_progress：进度记录表：self.prefill_progress: dict[str, tuple[Request, int, float]] = {}
                    # 每个请求记录三个信息
                    # old_req	上一次记录的 Request 对象
                    # old_count	上一次观察到的已计算 Token 数
                    # last_progress	最近一次取得进展的时间
                    # self.prefill_progress = {
                    #     "request_001": (req1, 4096, 100.0),request_001 上次已经计算 4096 Token，最后进展时间是 100 秒。
                    #     "request_002": (req2, 2048, 102.0),
                    # }
                    # self.prefill_progress.get(req_id, 默认值)，如果没有记录就用默认值
                if old_req is not req or req.num_computed_tokens > old_count:
                        last_progress = now  # 更新这个请求的进展时间
                # 更新请求的进度记录
                self.prefill_progress[req_id] = (
                    req,
                    req.num_computed_tokens,
                    last_progress,
                )
                    # 计算 Prefill 请求中最长的停滞时间
                oldest_stall_ms = max(oldest_stall_ms, (now - last_progress) * 1000)
                    # 为什么取最大值？
                    # 为控制器可能需要知道：
                    # 当前是否存在一个 Prefill 请求已经长时间没有得到计算资源？
            elif req.status == RequestStatus.RUNNING:
                decodes += 1
                    # 清理不再需要跟踪的prefill请求
                    # self.prefill_progress.keys()：表示历史记录中所有请求的 ID。
                    # live：表示当前仍然处于 Prefill 阶段的请求 ID。
        for req_id in self.prefill_progress.keys() - live:
            del self.prefill_progress[req_id]
        return prefills, decodes, oldest_stall_ms

    # 动态预算如何接入原生 Scheduler
    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        # _observe()返回：(3, 8, 2500.0)表示：3 个 Prefill 请求，8 个 Decode 请求，最长 Prefill 停滞时间 2500ms
        budget = self.budget_controller.decide(*self._observe(time.monotonic()))
        capacity = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = budget # 临时修改调度预算
        #注意，这里并没有修改：self.scheduler_config.max_num_batched_tokens
        #也没有修改用户启动服务时的配置文件。
        # 而是临时修改 Scheduler 实例上的运行时属性：self.max_num_scheduled_tokens
        try:
            return super().schedule(*args, **kwargs) # 将自适应预算接入原生 Scheduler
        #调用父类 Scheduler 的 schedule()
        # 自定义 Scheduler 不重新实现请求选择、KV Cache 分配、Prefill/Decode 调度等原生逻辑，而是改变原生调度逻辑使用的预算。
        finally:
            self.max_num_scheduled_tokens = capacity #调度完成后恢复原始预算，下一轮重新计算

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self.prefill_progress.clear()