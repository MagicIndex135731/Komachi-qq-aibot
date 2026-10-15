"""Shared foreground access to own durable public actions, without chat history."""
from app.agent.tool_agent import AgentTool, ToolOutcome


def build_self_activity_tool(repository, blocking, guard):
    async def read(args):
        guard.require_valid()
        try:
            result = await blocking.run(repository.self_publications, date=args.get('date'))
        except ValueError:
            guard.require_valid()
            return ToolOutcome({'error': 'self_activity_date_invalid'}, 'blocked')
        guard.require_valid()
        return ToolOutcome(result)

    return AgentTool('read_self_activity',
        '查询自己今天或指定上海日期发没发QQ空间、真实回执和时间；成功/失败/未知分别报告。'
        '只查询本系统发布记录，不包含好友评论、群私对话；无记录不证明QQ手动发布不存在。'
        '问自己的发布事实时调用此工具，不根据巡检过程或会话记忆猜测；不会发布或重试。',
        {'type': 'object', 'properties': {'date': {'type': 'string', 'maxLength': 10,
            'description': 'today或YYYY-MM-DD；省略为当前上海日期'}},
         'required': [], 'additionalProperties': False}, read)
