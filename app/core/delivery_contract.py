"""QQ delivery-state prompt notes shared by conversation surfaces."""

QQ_BLOCKED_REPLY_NOTICE = "刚刚的回复可能包含敏感信息，被 QQ 拦截了，无法发送。"

QQ_BLOCKED_CONTEXT_NOTE = (
    "[系统投递状态：以上回复未在 QQ 群中送达；连续发送后仍被 QQ 拦截，可能包含敏感信息。"
    "后续回答不得复述其中的敏感细节，只能概括说明上一条回复可能包含敏感信息、无法详细发送。]"
)

DELIVERY_UNCERTAIN_CONTEXT_NOTE = (
    "[系统投递状态：上一条回复的发送结果未得到确认；系统不会自动重发，以避免重复消息。]"
)
