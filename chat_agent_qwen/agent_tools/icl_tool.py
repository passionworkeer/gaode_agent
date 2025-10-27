# agent_tools/icl_tool.py
from typing import List, Dict, Any
from .tools import BaseTool, ToolParameter
from ..ICL_agent.icl_agent import ICLAgent
import logging

logger = logging.getLogger(__name__)

class ICLTool(BaseTool):
    """
    一个封装了 ICL (In-Context Learning) Agent 的工具。
    当需要基于成功案例进行推理或生成类似解决方案时，可以使用此工具。
    """
    def __init__(self, icl_agent: ICLAgent):
        super().__init__(
            name="in_context_learning_search",
            description="在历史成功案例库中搜索与当前问题相似的解决方案，并利用这些案例来生成新的、更优的回答。适用于需要借鉴过往经验的复杂规划任务，例如'给我一个类似上次那样的三亚家庭游规划'。"
        )
        self.icl_agent = icl_agent

    def define_parameters(self) -> List[ToolParameter]:
        """定义工具所需的参数。"""
        return [
            ToolParameter(
                name="query",
                type="str",
                description="用户的原始查询或需要解决的问题描述。",
                required=True
            )
        ]

    async def arun(self, params: Dict[str, Any]) -> str:
        """
        异步执行 ICL 查找和生成。
        
        Args:
            params: 包含 'query' 的字典。
            
        Returns:
            由 ICL Agent 生成的、参考了相似案例的回答。
        """
        query = params.get("query")
        if not query:
            return "错误：使用 'in_context_learning_search' 工具时，必须提供 'query' 参数。"

        try:
            logger.info(f"🧠 正在执行 ICL 搜索，查询: {query}")
            # ICLAgent 的核心方法是 find_similar_and_generate
            # 它需要一个包含 'role' 和 'content' 的消息列表作为输入
            messages = [{"role": "user", "content": query}]
            response = await self.icl_agent.find_similar_and_generate(messages)
            logger.info("✅ ICL 搜索和生成成功。")
            return response
        except Exception as e:
            logger.error(f"❌ ICL 工具执行失败: {e}", exc_info=True)
            return f"在相似案例库中搜索时发生错误: {str(e)}"

    def execute(self, params: Dict[str, Any]) -> str:
        """同步执行的包装器（不推荐）"""
        import asyncio
        try:
            return asyncio.run(self.arun(params))
        except Exception as e:
            return f"执行 ICL 工具时出错: {e}"
