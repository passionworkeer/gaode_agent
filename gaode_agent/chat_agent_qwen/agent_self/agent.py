# chat_agent_qwen/agent_self/agent.py
import asyncio
from .chat_agent_qwen_3_max import QwenModel
from ..agent_tools.tools import (
    BaseTool,
    SecureCodeInterpreterTool,
    FileRunnerTool,
    TavilySearchTool,
    VisualizationTool,
    FileTool,
)
from ..agent_tools.rag_tool import RAGTool
from ..agent_tools.icl_tool import ICLTool
from ..ICL_agent.icl_agent import ICLAgent
from ..agent_mcp.agent_mcp_gaode import MCPClient
from ..agent_memory.memory import MemoryManager
from ..prompts.system_prompts import TASK_PLANNER_SYSTEM_PROMPT, TOOL_USAGE_GUIDELINES
from ..utils.json_parser import RobustJSONParser
from ..utils.message_validator import MessageValidator
from ..utils.step_context import StepContext, TaskStep, ExecutionStrategy
import logging
from enum import Enum
from pydantic import BaseModel, Field, create_model
import re
import os
import json
from typing import List, Dict, Any, Tuple, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class Intent(Enum):
    """用户意图的枚举类型"""
    GENERAL_CHAT = "general_chat"
    TOOL_INFO_QUERY = "tool_info_query"
    KNOWLEDGE_QUERY_ICL = "knowledge_query_icl"
    KNOWLEDGE_QUERY_RAG = "knowledge_query_rag"
    COMPLEX_TASK = "complex_task"


class Agent:
    def __init__(self, model: QwenModel, memory: MemoryManager):
        self.model = model
        self.memory = memory
        self.file_runner = FileRunnerTool()
        self.mcp_client = MCPClient()
        self.visualization_tool = VisualizationTool(llm_model=model, memory_manager=memory)
        self.file_tool = FileTool(llm_model=model, memory_manager=memory)
        self.icl_agent = ICLAgent(model)
        self.icl_tool = ICLTool(self.icl_agent)
        self.tools: Dict[str, BaseTool] = {}
        self.tool_methods: Dict[str, Coroutine] = {} # ✅ 新增：存储 MCP 工具方法
        
    async def _register_tools(self) -> None:
        """注册所有工具（包括MCP工具）"""
        # ✅ 步骤 1: 获取 MCP 工具的方法和元数据
        mcp_tool_methods = await self.mcp_client.get_tool_methods()
        mcp_tools_metadata = await self.mcp_client.get_tools_metadata()

        # ✅ 直接将MCP方法注册到tool_methods（用于执行）
        self.tool_methods.update(mcp_tool_methods)
        logger.info(f"✅ 加载了 {len(mcp_tool_methods)} 个 MCP 工具方法: {list(mcp_tool_methods.keys())}")

        # ✅ 步骤 2: 将 MCP 元数据转换为 BaseTool 兼容对象以便统一描述
        mcp_tools_as_basetools = {}
        for meta in mcp_tools_metadata:
            properties_dict = meta.get('parameters', {}).get('properties', {})
            
            # 动态创建 Pydantic 模型作为 args_schema
            args_fields = {}
            for field_name, field_definition in properties_dict.items(): 
                mcp_type = field_definition.get("type", "string")
                python_type = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                }.get(mcp_type, str)
                
                description = field_definition.get("description", "")
                args_fields[field_name] = (python_type, Field(..., description=description))

            dynamic_args_model = create_model(
                f"{meta['name']}Args",
                **args_fields
            ) if args_fields else None

            # 创建 BaseTool 兼容的类
            tool_class = type(
                meta['name'],
                (BaseTool,),
                {
                    'name': meta['name'],
                    'description': meta['description'],
                    'args_schema': dynamic_args_model,
                }
            )
            mcp_tools_as_basetools[meta['name']] = tool_class(
                name=meta['name'], 
                description=meta['description']
            )

        # ✅ 步骤 3: 合并所有工具描述对象（用于LLM查看）
        self.tools = {
            "web_search": TavilySearchTool(),
            "secure_python_interpreter": SecureCodeInterpreterTool(),
            "run_created_file": self.file_runner,
            "rag_query": RAGTool(),
            "in_context_learning_search": self.icl_tool,
            "visualization_tool": self.visualization_tool,
            "file_tool": self.file_tool,
            **mcp_tools_as_basetools  # 合并 MCP 工具的描述对象
        }
        logger.info(f"✅ 工具描述已注册: {list(self.tools.keys())}")
        logger.info(f"✅ 工具执行方法已注册: {list(self.tool_methods.keys())}")


    def list_tools(self) -> str:
        """返回所有工具的描述字符串，供LLM使用。"""
        tool_strings = []
        for name, tool in self.tools.items():
            # 基础描述
            description = getattr(tool, 'description', 'No description available.')
            tool_str = f"- {name}: {description}"

            # 尝试获取并格式化参数
            if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_fields'):
                params = tool.args_schema.model_fields
                if params:
                    param_details = []
                    for param_name, field_info in params.items():
                        param_desc = getattr(field_info, 'description', '')
                        param_details.append(f"  - {param_name}: {param_desc}")
                    if param_details:
                        tool_str += "\n  参数:\n" + "\n".join(param_details)
            tool_strings.append(tool_str)
        
        return "\n".join(tool_strings)

    async def _classify_intent(self, user_input: str, history: List[Dict]) -> Intent:
        """使用LLM对用户意图进行分类"""
        prompt = f"""
[任务]
根据用户的最新指令，将其分类到以下意图之一：

1.  **general_chat**: 普通闲聊、问候、非功能性对话。
    (示例: "你好", "你叫什么名字?", "今天天气真好")
2.  **tool_info_query**: 询问关于Agent能力、可用工具的问题。
    (示例: "你能做什么?", "你有哪些工具?")
3.  **knowledge_query_icl**: 可以通过少量示例快速回答的知识性问题，通常是关于推荐或简单比较。
    (示例: "推荐深圳周末去哪玩?", "广州和上海哪个更适合旅游?")
4.  **knowledge_query_rag**: 需要从本地知识库中查找特定信息的问题。
    (示例: "深圳有哪些必去景点?", "介绍一下大鹏所城")
5.  **complex_task**: 需要执行多个步骤、调用一个或多个工具才能完成的复杂请求。
    (示例: "规划一个从深圳到北京的三日游", "帮我查一下从我家到公司怎么走，并把路线图画出来")

[历史对话]
{history[-3:]}

[用户最新指令]
"{user_input}"

[输出]
请仅返回最匹配的意图类别名称（例如: "complex_task"）。
"""
        response = await self.model.agenerate(prompt)
        intent_str = response.strip().lower()

        try:
            return Intent(intent_str)
        except ValueError:
            logger.warning(f"未知的意图: '{intent_str}', 降级为 'complex_task'")
            return Intent.COMPLEX_TASK

    async def _need_tool_use(self, user_input: str, history: List[Dict]) -> bool:
        """(异步)判断是否需要使用工具。"""
        # 如果工具列表为空，先注册
        if not self.tools:
            await self._register_tools()

        prompt = f"""
[可用工具]
{self.list_tools()}

[历史对话]
{history[-5:]}

[用户最新指令]
"{user_input}"

[判断任务]
根据用户的最新指令，判断是否必须使用上述一个或多个工具才能完成。
- 如果用户在进行常规聊天、打招呼、问候、表达观点，而没有提出具体的操作性需求，则回答 "否"。
- 如果用户的指令明确要求或暗示了需要进行搜索、计算、查询、画图、文件操作等，则回答 "是"。
- 如果用户的指令是关于代码或执行代码，则回答 "是"。
- 如果用户的指令是查询关于工具本身能做什么，则回答 "否"。

请只回答 "是" 或 "否"。
"""
        # 使用异步生成方法
        response = await self.model.agenerate(prompt)
        decision = response.strip()
        logger.info(f"工具使用决策: '{decision}' (原始输出: '{response}')")
        return "是" in decision

    async def plan_tasks(self, user_input: str, user_id: str) -> List[TaskStep]:
        """异步任务规划（确保工具已加载）"""
        memory = self.memory.load_memory(user_id)
        history = memory["conversation_history"]

        if not self.tools:
            await self._register_tools()

        # ✅ 使用 list_tools() 生成更简洁、结构化的工具列表
        system_content = (
            f"{TASK_PLANNER_SYSTEM_PROMPT}\n\n"
            f"{TOOL_USAGE_GUIDELINES}\n\n"
            f"可用工具列表:\n{self.list_tools()}"
        )
        
        prompt = [
                        {"role": "system", "content": system_content},
                        {"role": "system", "content": """输出格式: 严格 JSON 数组,每个对象仅包含 goal、tool_name、parameters 三个字段。

绝对约束(必须遵守,否则执行会失败):
1) 工具名必须为具体的官方名称,例如: maps_geo, maps_direction_driving, maps_direction_walking, maps_direction_bicycling, maps_direction_transit_integrated, maps_text_search, maps_around_search, maps_search_detail, maps_weather, maps_regeocode, maps_ip_location, maps_distance, visualization_tool, file_tool, web_search, rag_query, in_context_learning_search。
     - 严禁使用任何中介别名(如 mcp_tool)。不要输出 {\"tool_name\": \"mcp_tool\"} 这样的结构。
     - 严禁在 parameters 内再次嵌套 {\"tool_name\": ..., \"parameters\": ...}。
2) 占位符仅允许字段/下标访问,格式: {step_N_result.field} 或 {step_N_result.array[0].field}。
     - 严禁任何方法/函数调用(例如 .split(), .get(), int(), float() 等)与任何算术/比较运算(+, -, *, /, %, >, < 等)。
     - 如果需要坐标,高德返回结果已提供独立字段: lng(经度,浮点数), lat(纬度,浮点数)。请直接引用:
             • {step_0_result.geocodes[0].lng} / {step_0_result.geocodes[0].lat}
             • {step_1_result.pois[0].lng} / {step_1_result.pois[0].lat}
         若工具需要完整坐标字符串(如 maps_direction_* 的 origin/destination),请引用已有的 location 字段:
             • {step_0_result.geocodes[0].location} 或 {step_1_result.pois[0].location}
3) 步骤索引从 0 开始,只能引用之前步骤的结果(不得引用当前或未来步骤)。

正确示例:
[
    {"goal": "获取深圳技术大学经纬度", "tool_name": "maps_geo", "parameters": {"address": "深圳技术大学"}},
    {"goal": "规划驾车路线", "tool_name": "maps_direction_driving", "parameters": {"origin": "{step_0_result.geocodes[0].location}", "destination": "114.029963,22.609185"}},
    {"goal": "生成地图", "tool_name": "visualization_tool", "parameters": {"type": "map", "data": {"title": "路线地图", "markers": [{"lng": {step_0_result.geocodes[0].lng}, "lat": {step_0_result.geocodes[0].lat}, "title": "深圳技术大学"}]}}}
]

错误示例(禁止):
[
    {"tool_name": "mcp_tool", "parameters": {"tool_name": "maps_geo", "parameters": {"address": "深圳"}}},
    {"tool_name": "visualization_tool", "parameters": {"data": {"markers": [{"lng": "{step_0_result.pois[0].location.split(',')[0]}", "lat": "{step_0_result.pois[0].location.split(',')[1]}"}]}}}
]
"""},
                ]

        prompt = MessageValidator.safe_extend_history(prompt, history, max_count=3)
        prompt.append({"role": "user", "content": user_input})

        validated_prompt = MessageValidator.validate_messages(prompt)
        response = self.model.generate(validated_prompt).strip()
        
        steps_data = RobustJSONParser.parse(response)

        if not steps_data or not isinstance(steps_data, list):
            logger.warning(f"任务规划失败: 无法解析为列表 | 输出: {response[:200]}...")
            return []

        try:
            return [TaskStep(**s) for s in steps_data]
        except Exception as e:
            logger.warning(f"任务步骤解析失败: {e} | 数据: {steps_data}")
            return []

    async def execute_step(self, step: TaskStep, user_id: str) -> Tuple[bool, Any]:
        """执行单个步骤，支持自定义 MCP 工具方法和标准工具"""
        
        # ✅ 调试日志：打印当前步骤和可用工具
        logger.info(f"🔍 准备执行步骤: {step.tool_name}")
        logger.debug(f"📋 可用MCP工具方法: {list(self.tool_methods.keys())}")
        logger.debug(f"📋 可用标准工具: {list(self.tools.keys())}")
        
        # ⚠️ 友好拦截: 若仍出现历史别名 mcp_tool, 直接给出明确错误与指导
        if step.tool_name == "mcp_tool":
            guidance = (
                "检测到无效工具名 'mcp_tool'。请直接使用具体的官方工具名, 例如: "
                f"{', '.join(sorted(list(self.tool_methods.keys()))[:8])} ...。"
                "不要在 parameters 内再次嵌套 tool_name/parameters; 按 {tool_name, parameters} 直接提供。"
            )
            logger.error(guidance)
            return False, {"error": guidance}
        
        # ✅ 优先检查是否为 MCP 工具方法
        if step.tool_name in self.tool_methods:
            target_callable = self.tool_methods[step.tool_name]
            logger.info(f"🛠️ 执行 MCP 工具方法: {step.goal} | 工具: {step.tool_name} | 参数: {step.parameters}")
            try:
                result = await target_callable(**step.parameters)
                logger.info(f"✅ MCP 工具 '{step.tool_name}' 执行成功")
                return True, result
            except Exception as e:
                error_msg = f"MCP 工具 '{step.tool_name}' 执行错误: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return False, {"error": error_msg}

        # 检查标准工具
        tool = self.tools.get(step.tool_name)
        if not tool:
            all_available = list(self.tools.keys()) + list(self.tool_methods.keys())
            error_msg = f"❌ 工具 '{step.tool_name}' 不存在。可用工具: {all_available}"
            logger.error(error_msg)
            return False, error_msg

        try:
            logger.info(f"🛠️ 执行标准工具: {step.goal} | 工具: {step.tool_name} | 参数: {step.parameters}")
            
            # --- ✅ 修改开始：确保 params 是字典，并注入 user_id ---
            original_params = step.parameters
            # 确保 params 是一个字典，以便我们可以安全地添加 user_id 等键值对
            if isinstance(original_params, dict):
                # 使用 copy() 避免修改原始 step.parameters (虽然通常不是必需的，但更安全)
                params = original_params.copy() 
            else:
                # 如果 params 不是字典 (例如 None, str, list 等)
                # 1. 记录警告，因为这可能不是预期的行为
                logger.warning(
                    f"步骤 '{step.tool_name}' 的参数不是字典类型 (类型: {type(original_params)})。"
                    f"将尝试将其作为 'input' 参数传递给工具，并为需要的工具注入 user_id。"
                )
                # 2. 创建一个新的字典来存放参数
                #    - 将原始参数 (如果不是 None) 放入 'input' 键下。
                #    - 这是一种通用的处理方式，但具体工具可能需要不同的处理逻辑。
                #    - 对于 visualization_tool 和 file_tool，它们期望的是扁平的键值对参数，
                #      所以我们主要关心 user_id 的注入。如果原始参数很重要，
                #      工具内部需要能处理 'input' 键，或者这里需要更复杂的逻辑。
                params = {}
                if original_params is not None:
                    params["input"] = original_params # 可根据工具约定调整键名

            # 为特定工具注入 user_id（带兜底：CURRENT_USER_ID 或 anonymous）
            if step.tool_name in ["visualization_tool", "file_tool"]:
                effective_uid = (user_id or os.environ.get("CURRENT_USER_ID") or "anonymous")
                if not params.get("user_id"):
                    params["user_id"] = effective_uid
                    logger.debug(f"已为工具 '{step.tool_name}' 注入 user_id: {effective_uid}")

            # 处理 ICL tool 的特殊情况 (如果适用)
            # 注意：如果 original_params 不是字典，这可能不适用或需要调整
            if step.tool_name == "in_context_learning_search" and "query" not in params:
                 # 注意：如果 step.goal 不是字符串或不适合做 query，这里可能需要调整
                params["query"] = getattr(step, 'goal', '') # 使用 getattr 避免 AttributeError
            # --- ✅ 修改结束 ---

            # --- ✅ 修改：调用工具 ---
            # 现在 params 肯定是字典了，可以安全地使用 **kwargs 解包
            if hasattr(tool, 'arun'):
                raw_result = await tool.arun(**params) # type: ignore
            else:
                # 注意：如果 params 不是工具 run 方法期望的类型（例如，它是一个字典，
                # 但工具 run 期望一个字符串或位置参数），这可能会失败。
                # 理想情况下，所有工具都应该统一使用 arun/run 并接受字典参数。
                # 这里我们保持原逻辑，但如果 params 结构复杂，可能需要更细致的处理。
                raw_result = tool.run(**params)
            # --- ✅ 修改结束 ---

            # --- ✅ 修改：结果处理 ---
            # 标准工具可能返回字符串，尝试解析为 JSON
            # 也有可能直接返回 Python 对象 (dict, list 等)
            if isinstance(raw_result, str):
                try:
                    # 尝试解析为 JSON 对象
                    parsed_result = json.loads(raw_result)
                except (json.JSONDecodeError, TypeError):
                    # 如果解析失败，保留原始字符串
                    logger.debug(f"工具 '{step.tool_name}' 返回的字符串无法解析为JSON，将作为原始字符串处理。")
                    parsed_result = raw_result
            else:
                # 如果不是字符串，直接使用返回值
                parsed_result = raw_result

            logger.info(f"✅ 标准工具 '{step.tool_name}' 执行成功")
            return True, parsed_result
            # --- ✅ 修改结束 ---
            
        except Exception as e:
            error_msg = f"标准工具 '{step.tool_name}' 执行错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, {"error": error_msg}

    async def run(
        self, 
        user_input: str, 
        user_id: str = "default", 
        use_icl: bool = False,
        strategy: ExecutionStrategy = ExecutionStrategy.GRACEFUL_DEGRADE,
        stream_callback: Optional[Callable[[str], Coroutine]] = None
    ):
        """执行Agent主流程（异步生成器，支持流式输出）
        
        Args:
            user_input: 用户输入
            user_id: 用户ID
            use_icl: 是否使用ICL
            strategy: 执行策略 (失败处理方式)
            stream_callback: 用于流式输出的异步回调函数（已废弃，现在直接 yield）
        
        Yields:
            str: 流式输出的文本片段
        """
        full_response = ""
        
        try:
            # ✅ 确保工具已异步注册（包括MCP）
            if not self.tools:
                await self._register_tools()
                logger.info(f"✅ 已注册 {len(self.tools)} 个工具")
            
            self.memory.update_history(user_id, {"role": "user", "content": user_input})
            memory = self.memory.load_memory(user_id)

            # 1. 意图分类
            intent = await self._classify_intent(user_input, memory["conversation_history"])
            chunk = f"🔍 意图分析完成: **{intent.value}**\n\n"
            full_response += chunk
            yield chunk

            # 2. 根据意图执行不同逻辑
            if intent == Intent.GENERAL_CHAT:
                # 构建用于普通聊天的 Prompt
                memory = self.memory.load_memory(user_id)
                history = memory["conversation_history"]
                
                # 使用 ICL Agent 的示例（如果启用且有示例）
                icl_examples = ""
                if use_icl and self.icl_agent.examples:
                    icl_examples = "\n\n".join([f"示例 {i+1}:\n用户: {ex['query']}\n助手: {ex['response']}" for i, ex in enumerate(self.icl_agent.examples)])
                
                system_prompt = f"""你是一个智能助手，可以与用户自然对话。
            {icl_examples}
            """
                # 构建 Messages
                messages = [{"role": "system", "content": system_prompt}]
                messages = MessageValidator.safe_extend_history(messages, history, max_count=5)
                messages.append({"role": "user", "content": user_input})
                validated_messages = MessageValidator.validate_messages(messages)

                # 调用模型流式生成
                response_generator = self.model.stream_generate(validated_messages) 
                
                # 流式输出
                if hasattr(response_generator, '__aiter__'):
                    async for chunk in response_generator:
                        full_response += chunk
                        yield chunk
                elif hasattr(response_generator, '__iter__'):
                    for chunk in response_generator:
                        full_response += chunk
                        yield chunk
                else:
                    # 如果不是生成器，直接输出
                    chunk = str(response_generator)
                    full_response += chunk
                    yield chunk
                    
                self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                return

            if intent == Intent.TOOL_INFO_QUERY:
                reply = "我具备以下能力：\n" + self.list_tools()
                full_response += reply
                yield reply
                self.memory.update_history(user_id, {"role": "assistant", "content": reply})
                return

            if intent == Intent.KNOWLEDGE_QUERY_ICL:
                chunk = "好的，我将使用我的知识库为您快速解答...\n\n"
                full_response += chunk
                yield chunk
                tool = self.tools["in_context_learning_search"]
                result = await tool.arun(query=user_input)
                result_str = str(result)
                full_response += result_str
                yield result_str
                self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                return

            if intent == Intent.KNOWLEDGE_QUERY_RAG:
                chunk = "正在查询本地知识库...\n\n"
                full_response += chunk
                yield chunk
                tool = self.tools["rag_query"]
                result = await tool.arun(query=user_input)
                result_str = str(result)
                full_response += result_str
                yield result_str
                self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                return

            # --- 默认执行复杂任务逻辑 ---
            chunk = "好的，请稍等，我正在思考如何处理您的请求...\n\n"
            full_response += chunk
            yield chunk

            steps = await self.plan_tasks(user_input, user_id)
            if not steps:
                reply = "抱歉，我无法为您的请求规划出有效的执行步骤。请尝试换一种方式提问，或者描述得更具体一些。"
                full_response += reply
                yield reply
                self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                return

            chunk = "我已经制定了如下计划：\n"
            full_response += chunk
            yield chunk
            for i, step in enumerate(steps):
                chunk = f"   - 步骤 {i+1}: {step.goal}\n"
                full_response += chunk
                yield chunk
            chunk = "\n现在，我将开始执行这些步骤...\n\n"
            full_response += chunk
            yield chunk

            step_context = StepContext()
            steps_results = []
            
            for i, step in enumerate(steps):
                chunk = f"**正在执行步骤 {i+1}: {step.goal}**\n"
                full_response += chunk
                yield chunk
                try:
                    resolved_params = step_context.replace_placeholders(step.parameters, i)
                    resolved_step = TaskStep(goal=step.goal, tool_name=step.tool_name, parameters=resolved_params)
                    logger.info(f"步骤 {i}: {step.goal} | 解析后参数: {resolved_params}")
                    chunk = f"   - 调用工具: `{step.tool_name}`\n"
                    full_response += chunk
                    yield chunk
                    chunk = f"   - 提供参数: `{json.dumps(resolved_params, ensure_ascii=False, indent=2)}`\n"
                    full_response += chunk
                    yield chunk
                except ValueError as e:
                    error_msg = f"参数解析失败: {e}"
                    logger.error(f"步骤 {i} {error_msg}")
                    chunk = f"   - ❌ **错误**: {error_msg}\n"
                    full_response += chunk
                    yield chunk
                    if strategy == ExecutionStrategy.FAIL_FAST:
                        final_reply = f"抱歉，任务在'{step.goal}'步骤中断，因为参数准备失败。"
                        full_response += f"\n{final_reply}"
                        yield f"\n{final_reply}"
                        self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                        return
                    success, result = False, {"error": error_msg}
                    resolved_step = step
                else:
                    success, result = await self.execute_step(resolved_step, user_id)
                
                if not success:
                    error_msg = f"步骤 {i+1} 执行失败: {result}"
                    logger.error(error_msg)
                    chunk = f"   - ❌ **错误**: {result}\n\n"
                    full_response += chunk
                    yield chunk
                    if strategy == ExecutionStrategy.FAIL_FAST:
                        reply = f"抱歉，在执行'{step.goal}'时遇到问题：{result}\n\n请尝试重新描述您的需求。"
                        full_response += reply
                        yield reply
                        self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                        return
                    elif strategy == ExecutionStrategy.GRACEFUL_DEGRADE:
                        logger.info(f"步骤失败，降级为纯对话模式")
                        chunk = "哎呀，执行计划遇到了一点小问题。我将尝试根据现有信息为您总结回答。\n"
                        full_response += chunk
                        yield chunk
                        break
                
                try:
                    if isinstance(result, str):
                        result_dict = RobustJSONParser.parse(result)
                        step_context.set_result(i, result_dict if result_dict and isinstance(result_dict, dict) else {"raw": result, "success": success})
                    else:
                        step_context.set_result(i, result if isinstance(result, dict) else {"raw": str(result), "success": success})
                    logger.info(f"✅ 步骤 {i} 结果已保存")
                    chunk = f"   - ✅ **成功**: 步骤完成，结果已保存。\n\n"
                    full_response += chunk
                    yield chunk
                except Exception as parse_error:
                    logger.warning(f"⚠️ 步骤 {i} 结果保存失败: {parse_error}, 保存原始值")
                    step_context.set_result(i, {"raw": str(result), "success": success, "error": str(parse_error)})
                    chunk = f"   - ⚠️ **警告**: 步骤结果保存时遇到问题: {parse_error}\n\n"
                    full_response += chunk
                    yield chunk

                steps_results.append((resolved_step, result))
                self.memory.update_history(user_id, {"role": "system", "content": f"执行步骤 {i}: {step.goal}\n工具: {step.tool_name}\n结果: {json.dumps(result, ensure_ascii=False, indent=2)}"})

            chunk = "所有步骤执行完毕，现在我将为您整合最终结果...\n\n---\n\n"
            full_response += chunk
            yield chunk
            
            # 调用结果整合方法（异步生成器）
            async for chunk in self.integrate_results_stream(user_input, steps_results, user_id):
                full_response += chunk
                yield chunk

            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"Agent 主流程发生意外错误: {e}", exc_info=True)
            error_message = f"\n\n--- \n**系统错误** \n抱歉，我在处理您的请求时遇到了一个意外的问题: `{str(e)}` \n请稍后再试或联系技术支持。"
            full_response += error_message
            yield error_message
            # 确保即使在顶层异常中，最终的错误信息也被记录
            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
    
    async def integrate_results_stream(
        self, 
        user_input: str, 
        steps_results: List[Tuple[TaskStep, Any]], 
        user_id: str
    ):
        """整合所有步骤结果，生成最终回复（流式输出）
        
        Args:
            user_input: 用户原始输入
            steps_results: 所有步骤的执行结果列表 [(TaskStep, result), ...]
            user_id: 用户ID
        
        Yields:
            str: 生成的文本片段
        """
        try:
            # ✅ 步骤 1: 提取关键信息
            extracted_data = self._extract_key_information(steps_results)
            
            # ✅ 步骤 2: 构建整合提示词
            from ..prompts.system_prompts import RESULT_INTEGRATION_SYSTEM_PROMPT
            
            # 将步骤结果格式化为可读文本
            steps_summary = []
            for i, (step, result) in enumerate(steps_results):
                steps_summary.append(f"步骤 {i+1}: {step.goal}")
                steps_summary.append(f"工具: {step.tool_name}")
                
                # 格式化结果
                if isinstance(result, dict):
                    result_str = json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    result_str = str(result)
                
                steps_summary.append(f"结果: {result_str[:500]}...")  # 限制长度
                steps_summary.append("---")
            
            steps_text = "\n".join(steps_summary)
            
            # 获取历史对话
            memory = self.memory.load_memory(user_id)
            history = memory["conversation_history"]
            
            # ✅ 步骤 3: 构建增强的整合提示
            integration_prompt = self._build_integration_prompt(
                user_input, 
                steps_text, 
                extracted_data
            )
            
            # 构建消息
            messages = [
                {"role": "system", "content": RESULT_INTEGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": integration_prompt}
            ]
            
            # 添加部分历史上下文（最备2轮）
            messages = MessageValidator.safe_extend_history(messages, history, max_count=2)
            
            # 验证消息
            validated_messages = MessageValidator.validate_messages(messages)
            
            # 流式生成最终回复（同时收集正文用于后续判断是否需要兜底追加）
            generated_chunks: List[str] = []
            async for chunk in self.model.astream_generate(validated_messages):
                generated_chunks.append(str(chunk))
                yield chunk
            
            # ✅ 步骤 4: 在流式输出后追加资源：文件、地图、图片（仅在正文未包含时兜底追加）
            body_text = "".join(generated_chunks)
            has_appended_header = False

            # 4.1 追加文件链接
            if extracted_data.get("file_paths"):
                yield "\n\n---\n\n"
                has_appended_header = True
                yield "📄 **生成的文件**:\n\n"
                for file_info in extracted_data["file_paths"]:
                    file_type = file_info.get("type", "文件")
                    file_path = file_info.get("path", "")
                    if file_path:
                        yield f"- [{file_type}]({file_path})\n"
                        logger.info(f"✅ 添加文件链接: {file_path}")

            # 4.2 追加地图 HTML 链接
            if extracted_data.get("map_paths"):
                if not has_appended_header:
                    yield "\n\n---\n\n"
                    has_appended_header = True
                # 若正文中尚未包含地图链接，再追加兜底
                body_has_map = False
                for map_path in extracted_data["map_paths"]:
                    if isinstance(map_path, str) and map_path in body_text:
                        body_has_map = True
                        break
                if not body_has_map:
                    yield "🗺️ **地图**:\n\n"
                    for idx, map_path in enumerate(extracted_data["map_paths"], start=1):
                        if isinstance(map_path, str) and map_path.endswith(".html"):
                            yield f"- [打开地图 {idx}]({map_path})\n"
                            logger.info(f"✅ 添加地图链接: {map_path}")

            # 4.3 追加图片预览（Markdown 多图）
            if extracted_data.get("images"):
                if not has_appended_header:
                    yield "\n\n---\n\n"
                    has_appended_header = True
                # 若正文中已经包含图片（通过 Markdown ![]() 或包含已知 URL）则不再兜底
                body_has_inline_image = ("![" in body_text) or any(
                    isinstance(u, str) and u in body_text for u in extracted_data["images"]
                )
                if not body_has_inline_image:
                    yield "🖼️ **图片预览**:\n\n"
                    # 支持每个景点多张图片（假定 images 为所有图片，分组逻辑可后续增强）
                    for i, url in enumerate(extracted_data["images"][:10], start=1):
                        if isinstance(url, str) and url.startswith("http"):
                            yield f"- 图片 {i}: {url}\n"
                            yield f"![图片 {i}]({url})\n"
                            logger.info(f"✅ 兜底追加图片URL: {url}")
                
        except Exception as e:
            logger.error(f"结果整合失败: {e}", exc_info=True)
            yield f"\n\n⚠️ 整合结果时遇到问题: {str(e)}\n\n"
            yield "不过，根据执行的步骤，我可以告诉您：\n"
            
            # 降级方案：简单列出结果
            for i, (step, result) in enumerate(steps_results):
                yield f"\n{i+1}. {step.goal}: "
                if isinstance(result, dict) and "error" not in result:
                    yield "✅ 成功"
                else:
                    yield "⚠️ 部分完成"
    
    def _extract_key_information(self, steps_results: List[Tuple[TaskStep, Any]]) -> Dict[str, Any]:
        """从步骤结果中提取关键信息
        
        Args:
            steps_results: 所有步骤的执行结果
            
        Returns:
            包含提取信息的字典
        """
        extracted = {
            "weather_data": [],
            "file_paths": [],
            "map_paths": [],
            "routes": [],
            "pois": [],
            "images": [],
            "web_search_images": [],  # [{query, image_urls}]
            "poi_images": {}          # {poi_name: [urls]}
        }
        
        for step, result in steps_results:
            tool_name = step.tool_name
            
            # ✅ 提取天气数据
            if tool_name == "maps_weather" and isinstance(result, dict):
                if "forecasts" in result:
                    extracted["weather_data"].extend(result["forecasts"])
                    logger.info(f"✅ 提取天气数据: {len(result['forecasts'])} 条")
            
            # ✅ 提取文件路径
            elif tool_name == "file_tool" and isinstance(result, str):
                # file_tool 返回的是文件路径字符串
                if result and not result.startswith("错误"):
                    file_type = "行程文档"
                    if ".html" in result:
                        file_type = "HTML文档"
                    elif ".pdf" in result:
                        file_type = "PDF文档"
                    elif ".xlsx" in result or ".xls" in result:
                        file_type = "Excel表格"
                    
                    extracted["file_paths"].append({
                        "type": file_type,
                        "path": result
                    })
                    logger.info(f"✅ 提取文件路径: {result}")
            
            # ✅ 提取地图路径
            elif tool_name == "visualization_tool" and isinstance(result, str):
                if result and result.endswith(".html") and not result.startswith("错误"):
                    extracted["map_paths"].append(result)
                    logger.info(f"✅ 提取地图路径: {result}")
            
            # ✅ 提取路线信息
            elif "direction" in tool_name and isinstance(result, dict):
                if "paths" in result or "route" in result:
                    extracted["routes"].append({
                        "tool": tool_name,
                        "data": result
                    })
                    logger.info(f"✅ 提取路线信息: {tool_name}")
            
            # ✅ 提取 POI 信息
            elif tool_name == "maps_text_search" and isinstance(result, dict):
                if "pois" in result:
                    extracted["pois"].extend(result["pois"][:5])  # 只取前5个
                    logger.info(f"✅ 提取 POI 信息: {len(result['pois'])} 个")
            
            # ✅ 提取搜索结果中的图片 URL
            elif tool_name == "web_search":
                # 1) 优先从结构化字段提取（推荐路径）
                if isinstance(result, dict):
                    image_urls = result.get("image_urls") or []
                    # 记录 query 与图片的映射，便于后续按景点绑定
                    try:
                        query_text = ""
                        if isinstance(step.parameters, dict):
                            query_text = step.parameters.get("query", "") or ""
                    except Exception:
                        query_text = ""
                    if isinstance(image_urls, list) and image_urls:
                        # 最多保留 5 张
                        urls = [u for u in image_urls[:5] if isinstance(u, str)]
                        extracted["images"].extend(urls)
                        if query_text:
                            extracted["web_search_images"].append({
                                "query": query_text,
                                "image_urls": urls
                            })
                        logger.info(f"✅ 从 web_search.image_urls 提取图片: {len(urls)} 张 | query='{query_text[:40]}'")
                    if isinstance(image_urls, list) and image_urls:
                        pass
                # 2) 兼容旧路径：从字符串结果中用正则提取
                elif isinstance(result, str):
                    import re
                    image_urls = re.findall(r'https?://[^\s<>"]+?\.(?:jpg|jpeg|png|gif|webp)', result)
                    if image_urls:
                        extracted["images"].extend(image_urls[:3])  # 最多3张
                        logger.info(f"✅ 从字符串结果提取图片 URL: {len(image_urls)} 个")

        # 基于 web_search 的 query 粗略将图片绑定到 POI（按名称包含关系）
        try:
            poi_names = []
            for poi in extracted.get("pois", []) or []:
                name = poi.get("name") or poi.get("title")
                if isinstance(name, str) and name:
                    poi_names.append(name)
            if poi_names and extracted.get("web_search_images"):
                for pair in extracted["web_search_images"]:
                    q = (pair.get("query") or "").lower()
                    urls = pair.get("image_urls") or []
                    if not q or not urls:
                        continue
                    # 找到名称与 query 互为包含的 POI
                    target = None
                    for pn in poi_names:
                        pn_l = pn.lower()
                        if pn_l in q or q in pn_l:
                            target = pn
                            break
                    if target:
                        bucket = extracted["poi_images"].setdefault(target, [])
                        for u in urls:
                            if isinstance(u, str) and u not in bucket:
                                bucket.append(u)
        except Exception as e:
            logger.warning(f"⚠️ 绑定图片到 POI 映射时发生问题: {e}")
        
        return extracted
    
    def _build_integration_prompt(
        self, 
        user_input: str, 
        steps_text: str, 
        extracted_data: Dict[str, Any]
    ) -> str:
        """构建增强的整合提示
        
        Args:
            user_input: 用户原始输入
            steps_text: 步骤执行结果文本
            extracted_data: 提取的关键信息
            
        Returns:
            整合提示词
        """
        prompt_parts = [
            f"[用户原始问题]",
            user_input,
            "",
            f"[执行步骤及结果]",
            steps_text,
            "",
            f"[任务]",
            "请基于以上步骤的执行结果，生成一个连贯、自然、信息完整且可直接渲染的 Markdown 回复。",
            "",
            f"**核心要求**:"
        ]
        
        # ✅ 添加天气数据要求
        if extracted_data.get("weather_data"):
            weather_info = extracted_data["weather_data"][0]  # 取第一条
            prompt_parts.extend([
                f"1. **天气信息**: 使用以下真实数据：",
                f"   - 日期: {weather_info.get('date', 'N/A')}",
                f"   - 白天天气: {weather_info.get('dayweather', 'N/A')}",
                f"   - 夜间天气: {weather_info.get('nightweather', 'N/A')}",
                f"   - 白天温度: {weather_info.get('daytemp', 'N/A')}°C",
                f"   - 夜间温度: {weather_info.get('nighttemp', 'N/A')}°C",
                f"   ⚠️ **不要使用模糊的推测性语言，直接引用上述真实数据**",
                ""
            ])
        
        # ✅ 添加 POI 信息要求
        if extracted_data.get("pois"):
            prompt_parts.append(f"2. **景点信息**: 使用以下搜索到的真实 POI 数据，包含名称、地址等；对每个关键景点生成独立小节（含名称为小标题），避免纯列表堆叠。")
        
        # ✅ 添加图片要求（内联插入）
        if extracted_data.get("poi_images"):
            # 已有按景点分组的图片
            prompt_parts.extend([
                "3. 图片展示（必须内联）:",
                "   - 已为以下景点分组整理出真实图片 URL，请在对应景点小节末尾内联插入 1-3 张图片（优先使用每景点前 3 张）:",
            ])
            for poi_name, urls in list(extracted_data.get("poi_images", {}).items())[:8]:
                prompt_parts.append(f"   - {poi_name}:")
                for u in urls[:3]:
                    prompt_parts.append(f"     • {u}")
            prompt_parts.extend([
                "   - 图片 Markdown 语法示例：![](https://example.com/a.jpg)",
                "   - ❌ 绝对禁止: 使用示例或占位链接",
                "   - ✅ 正确做法: 每个景点段落的文字描述后紧跟一行插入对应图片",
                ""
            ])
        elif extracted_data.get("images"):
            prompt_parts.extend([
                "3. **图片展示（必须内联）**:\n"
                "   - 使用下列真实图片 URL，并将图片内联插入到相应段落（例如每个景点段落的末尾）中；不要把图片集中到文末。\n"
                "   - 优先为前 3~5 个关键景点各插入 1 张图片；若图片数不足，可为最重要的景点插图。\n"
                "   - 图片 Markdown 语法示例：![](https://example.com/a.jpg)\n"
                "   - 以下可用图片 URL（按顺序匹配景点）：",
                *[f"   - {url}" for url in extracted_data["images"][:5]],
                "   ❌ **绝对禁止**: 使用 example.com、placeholder.jpg 等示例链接",
                "   ✅ **正确做法**: 在每个景点小节的文字描述后紧跟一行插入对应图片",
                ""
            ])
        else:
            prompt_parts.extend([
                f"3. **图片处理**: 当前无可用图片,请在相关段落明确说明“暂无图片”，不要使用虚构链接",
                ""
            ])

        # 提供 web_search 的查询与图片列表，便于模型做更精确绑定（即使上面已有 poi_images）
        if extracted_data.get("web_search_images"):
            prompt_parts.extend([
                "附：图片检索明细（用于对齐景点名称）:",
            ])
            for pair in extracted_data.get("web_search_images", [])[:8]:
                q = pair.get("query", "")
                urls = pair.get("image_urls", [])
                prompt_parts.append(f"- 查询: {q}")
                for u in urls[:3]:
                    prompt_parts.append(f"  • {u}")
            prompt_parts.append("")
        
        # ✅ 添加路线信息要求
        if extracted_data.get("routes"):
            route_data = extracted_data["routes"][0].get("data", {})
            if "paths" in route_data and route_data["paths"]:
                path = route_data["paths"][0]
                prompt_parts.extend([
                    f"4. **路线信息** (必须使用精确数值):",
                    f"   - 总距离: {path.get('distance_km', 'N/A')} 公里",
                    f"   - 预计时间: {path.get('duration_min', 'N/A')} 分钟",
                    f"   ❌ **绝对禁止**: 使用“大约X公里”等模糊语言",
                    ""
                ])

        # ✅ 如果存在地图 HTML，要求在正文中合适位置内联一个链接
        if extracted_data.get("map_paths"):
            prompt_parts.extend([
                f"5. **地图链接（正文内联）**: 如果存在交互式地图 HTML，请在'路线'或'行程总览'段落中插入一个可点击链接，格式例如：[查看交互式地图](<map_path>)。",
                f"   可用地图路径（任选其一）：",
                *[f"   - {p}" for p in extracted_data.get("map_paths", [])[:2]],
                ""
            ])
        
        # ✅ 添加通用要求
        prompt_parts.extend([
            f"6. 回答用户的原始问题",
            f"7. 整合所有关键信息（路线、距离、时间、天气、景点等），并在相关段落就地内联插入图片和地图链接",
            f"8. 使用 Markdown 格式，结构清晰（使用二级/三级小标题）",
            f"9. 语言自然流畅，像真人回答一样",
            f"10. 文件路径可由系统在文末追加；但'地图'允许在正文中内联链接",
            "",
            f"**最后检查清单** (发送前必须确认):",
            f"☑️ 所有天气数据是否使用了maps_weather的真实返回值?",
            f"☑️ 是否已在相关段落内联插入图片（使用提供的 image_urls）?",
            f"☑️ 所有距离/时间是否使用了路线工具的精确返回值?",
            f"☑️ 如果存在地图 HTML，是否在正文合适位置插入了可点击链接?",
            f"☑️ 是否没有使用任何“可能”“大约”等不确定表述?"
        ])
        
        return "\n".join(prompt_parts)

