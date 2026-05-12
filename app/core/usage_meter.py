from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from datetime import UTC, datetime, tzinfo


GPT_54_INPUT_USD_PER_MILLION = Decimal("2.50")
GPT_54_CACHED_INPUT_USD_PER_MILLION = Decimal("0.25")
GPT_54_OUTPUT_USD_PER_MILLION = Decimal("15.00")


@dataclass(slots=True)
class UsageTotals:
    call_count: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int

    @property
    def regular_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)


def build_local_day_utc_window(
    current_time: datetime,
    *,
    local_timezone: tzinfo | None = None,
) -> tuple[datetime, datetime]:
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    effective_local_timezone = local_timezone or current_time.astimezone().tzinfo or UTC
    local_now = current_time.astimezone(effective_local_timezone)
    local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_start.astimezone(UTC), local_now.astimezone(UTC)


def estimate_gpt_54_cost_usd(totals: UsageTotals) -> Decimal:
    million = Decimal("1000000")
    regular_input_cost = (Decimal(totals.regular_input_tokens) / million) * GPT_54_INPUT_USD_PER_MILLION
    cached_input_cost = (Decimal(totals.cached_input_tokens) / million) * GPT_54_CACHED_INPUT_USD_PER_MILLION
    output_cost = (Decimal(totals.output_tokens) / million) * GPT_54_OUTPUT_USD_PER_MILLION
    return regular_input_cost + cached_input_cost + output_cost


def format_cost_usd(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def format_daily_admin_usage_report(totals: UsageTotals) -> str:
    cost_usd = estimate_gpt_54_cost_usd(totals)
    return (
        "今天截至你这次@我为止，"
        f"模型调用 {totals.call_count} 次，"
        f"输入 {totals.input_tokens} token，"
        f"其中缓存输入 {totals.cached_input_tokens} token，"
        f"输出 {totals.output_tokens} token，"
        f"按 OpenAI GPT-5.4 官方价格估算约 ${format_cost_usd(cost_usd)} USD。"
    )
