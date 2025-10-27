# 导入外部依赖
from ..RAG_agent.rag_engine import RAGEngine # 从相对路径导入 RAG 引擎
from .tools import BaseTool, ToolParameter # 从同目录下的 tools 模块导入基础类和参数定义类

# --- 工具类定义 ---
class RAGTool(BaseTool):
    """
    RAG (Retrieval-Augmented Generation) 工具类。

    该工具将用户查询与内部知识库进行匹配，检索出最相关的文档片段，
    并将其作为上下文提供给 Agent，以生成更准确、更基于事实的回答。
    适用于回答需要特定领域知识或内部信息的问题。
    """

    def __init__(self, rag_engine=None):
        """
        初始化 RAGTool 实例。

        Args:
            rag_engine (RAGEngine, optional): 一个已初始化的 RAGEngine 实例。
                                              如果未提供，则工具内部会创建一个新的 RAGEngine 实例。
                                              这允许外部传入共享的 RAG 引擎实例以优化资源。
        """
        # 调用父类 BaseTool 的初始化方法，设置工具名称和描述
        super().__init__(
            name="rag_query",  # 工具的唯一标识符，Agent 和 LLM 会使用此名称来调用工具
            description="基于 RAG 知识库的检索与问答工具。可用于查询特定领域知识、内部文档或 FAQ。" # 工具功能的简要描述，供 Agent 理解其用途
        )
        # 初始化 RAG 引擎实例
        # 如果外部传入了 rag_engine，则使用传入的实例；否则，创建一个新的 RAGEngine 实例
        self.rag_engine = rag_engine or RAGEngine()
        # TODO: (可选) 添加日志记录，记录 RAG 引擎初始化状态
        # logger.info("RAGTool initialized with RAGEngine instance.")

    def define_parameters(self):
        """
        定义工具执行时所需的参数及其规范。

        Returns:
            List[ToolParameter]: 一个包含 ToolParameter 对象的列表，描述了工具接受的参数。
        """
        # 定义工具可接受的参数列表
        return [
            # 必填参数：用户查询
            ToolParameter(
                name="query",  # 参数名称
                type="str",    # 参数类型
                description="用户提出的问题或查询语句。例如：'公司最新的差旅报销政策是什么？'", # 参数用途说明
                required=True  # 标记为必需参数
            ),
            # 可选参数：返回结果数量
            ToolParameter(
                name="top_k",      # 参数名称
                type="int",        # 参数类型
                description="指定返回最相关的知识片段数量，默认为 3。", # 参数用途说明
                required=False     # 标记为非必需参数，工具内部会使用默认值
            )
        ]

    def execute(self, params):
        """
        执行 RAG 查询的核心方法。

        该方法接收来自 Agent 的参数，调用 RAG 引擎进行查询，并处理返回结果。

        Args:
            params (Dict[str, Any]): 包含执行参数的字典，例如 {"query": "用户问题", "top_k": 5}。

        Returns:
            str: RAG 查询的结果，格式化为字符串。如果未找到结果，返回提示信息。
        """
        # 1. 参数提取与验证
        # 从传入的 params 字典中获取 'query' 参数，默认值为空字符串
        query = params.get("query", "")
        # 从传入的 params 字典中获取 'top_k' 参数，默认值为 3
        top_k = params.get("top_k", 3)

        # 2. 输入校验 (可选，但推荐)
        # 确保查询字符串不为空或仅包含空白字符
        if not query.strip():
            return "RAG 工具执行失败：'query' 参数不能为空。"

        # 3. 调用 RAG 引擎
        # 使用提取的 query 和 top_k 参数调用 RAG 引擎的 query 方法
        # 注意：此处假设 RAGEngine.query 方法是同步的。如果是异步的，需要调整实现。
        try:
            results = self.rag_engine.query(query, top_k=top_k)
        except Exception as e:
            # 4. 异常处理
            # 记录错误日志，包含用户ID (如果可获取)、查询内容和错误详情
            # logger.error(f"RAGTool query failed for query '{query}'. Error: {e}")
            return f"RAG 工具执行时发生错误: {str(e)}"

        # 5. 结果处理与返回
        # 如果 RAG 引擎返回了结果列表
        if results:
            # 将结果列表中的每个字符串片段用换行符连接成一个完整的字符串返回
            # 这样 Agent 可以将所有相关片段作为一个整体信息来理解
            formatted_results = "\n".join(results)
            return formatted_results
        else:
            # 如果 RAG 引擎返回空列表或 None，表示未找到相关知识
            return "未在知识库中检索到与您问题相关的信息。"
    
    async def arun(self, query: str, top_k: int = 3, **kwargs) -> str:
        """异步执行 RAG 查询
        
        Args:
            query: 用户查询
            top_k: 返回最相关的片段数量
            **kwargs: 其他参数
        
        Returns:
            str: 查询结果
        """
        # 复用 execute 方法的逻辑
        params = {"query": query, "top_k": top_k}
        params.update(kwargs)
        return self.execute(params)

