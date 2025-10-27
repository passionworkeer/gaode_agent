# tavily_search_tools.py
"""
Tavily Search 工具集 - 用于执行 Web 搜索和内容提取

此模块提供 TavilySearchTool 和 TavilyExtractTool，它们封装了 Tavily API 的功能。
通过 TavilySearchToolManager 进行统一管理和初始化。
"""

import os
import logging
from typing import List, Any, Optional, Dict, Literal
import re
from langchain_core.tools import BaseTool
from tavily import AsyncTavilyClient
from pydantic import Field, BaseModel

# 设置日志记录器
logger = logging.getLogger(__name__)

# --- 新增：定义 TavilySearchTool 和 TavilyExtractTool 的参数模型 ---
# 这有助于更好地定义工具的输入参数结构和进行类型检查


class TavilyExtractParams(BaseModel):
    """Tavily Extract 工具的参数模型"""
    urls: List[str] = Field(..., description="要提取内容的 URL 列表")


class TavilySearchParams(BaseModel):
    """Tavily Search 工具的参数模型"""
    query: str = Field(..., description="搜索关键词")
    search_depth: Literal["basic", "advanced"] = Field(default="advanced", description="搜索深度")
    max_results: int = Field(default=5, description="返回的最大结果数")
    days: Optional[int] = Field(default=None, description="限制搜索结果的天数")
    include_answer: bool = Field(default=True, description="是否包含 AI 生成的答案")
    include_images: bool = Field(default=True, description="是否包含图片链接")


class TavilySearchTool(BaseTool):
    """Tavily Web 搜索工具"""

    # --- ✅ 使用 Field 定义 name 和 description ---
    name: str = Field(default="tavily_search", description="工具名称")
    description: str = Field(
        default=(
            "Use this to perform a real-time web search and get a summarized answer with sources. "
            "Supports filtering by recency (days), depth (basic/advanced), and result count."
        ),
        description="工具描述"
    )
    
    # --- ✅ 移除错误的 args_schema 和 client 字段定义 ---
    # 不要在这里使用 Field 定义 args_schema 和 client
    # args_schema: TavilySearchParams = TavilySearchParams # ❌ 错误的定义
    # client: AsyncTavilyClient = Field(..., description="Tavily 异步客户端实例") # ❌ 不当的定义

    # --- ✅ 新增/修改：显式定义 __init__ 方法 ---
    def __init__(self, client: AsyncTavilyClient):
        """
        初始化 TavilySearchTool。
        
        Args:
            client (AsyncTavilyClient): 已初始化的 Tavily 异步客户端实例。
        """
        # --- ✅ 关键修改 1: 正确设置 args_schema 为 Pydantic 模型类 ---
        # 直接赋值给实例属性，不使用 Field
        self.args_schema = TavilySearchParams # ✅ 正确方式
        # --- ✅ 关键修改 2: 存储 client 实例 ---
        # 存储传入的 client 实例
        self.client = client # ✅ 正确方式
        # --- ✅ 关键修改 3: 调用父类 __init__ ---
        # 确保 BaseTool 正确初始化，传入必要的参数
        super().__init__( # type: ignore
            name="tavily_search",
            description=(
                "Use this to perform a real-time web search and get a summarized answer with sources. "
                "Supports filtering by recency (days), depth (basic/advanced), and result count."
            ),
            args_schema=TavilySearchParams, # 传入 args_schema 类
            client=client # 传入 client 实例 (虽然 BaseTool 不直接使用，但存储在 self.client)
        )
        # --- ✅ 修改结束 ---

    # --- ✅ 保留 _arun 方法 ---
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """
        异步执行 Tavily 搜索。

        Args:
            **kwargs: 搜索参数，应符合 TavilySearchParams 模型。
                      例如: query="最新科技新闻", search_depth="advanced", max_results=5

        Returns:
            Dict[str, Any]: Tavily API 返回的搜索结果字典，包含 answer, results, images 等。
                           增强功能：如果 include_images=True 且返回了图片，
                           会额外添加一个 'image_urls' 键，其值为提取出的图片 URL 列表。
                           
        Raises:
            ValueError: 如果参数验证失败。
            RuntimeError: 如果 Tavily API 调用失败。
        """
        # --- ✅ 使用 Pydantic 模型验证和解析参数 ---
        try:
            params = TavilySearchParams(**kwargs)
        except Exception as e:
            logger.error(f"TavilySearchTool 参数验证失败: {e}")
            raise ValueError(f"参数验证失败: {e}")

        # --- ✅ 构建安全的 kwargs（避免传入 None）---
        search_kwargs: Dict[str, Any] = {
            "query": params.query,
            "search_depth": params.search_depth,
            "include_answer": params.include_answer,
            "max_results": params.max_results,
            "include_images": params.include_images,
        }
        if params.days is not None:
            search_kwargs["days"] = params.days

        try:
            logger.debug(f"调用 Tavily API search: {search_kwargs}")
            response = await self.client.search(**search_kwargs) # type: ignore
            logger.debug(f"Tavily API search 响应类型: {type(response)}")
            if isinstance(response, dict):
                logger.debug(
                    "Tavily API search 响应概览: keys=%s, images_len=%s, results_len=%s",
                    list(response.keys()),
                    len(response.get("images", []) if isinstance(response.get("images", []), list) else []),
                    len(response.get("results", []) if isinstance(response.get("results", []), list) else []),
                )

            # --- ✅ 增强：多路径提取图片URL，并去重、裁剪 ---
            if isinstance(response, dict) and params.include_images:
                extracted: List[str] = []

                # 1) 顶层 images：可能是字符串 URL 列表，或对象列表
                raw_images = response.get("images", [])
                if isinstance(raw_images, list):
                    for img in raw_images:
                        if isinstance(img, str):
                            extracted.append(img)
                        elif isinstance(img, dict):
                            url = img.get("url") or img.get("image_url") or img.get("link")
                            if isinstance(url, str):
                                extracted.append(url)

                # 2) results[*] 内的 images / image_urls，以及文本字段中可能出现的图片链接
                results = response.get("results", []) or []
                img_url_regex = re.compile(r"https?://[^\s'\"<>]+?\.(?:jpg|jpeg|png|gif|webp)\b", re.IGNORECASE)
                if isinstance(results, list):
                    for item in results[:10]:  # 控制复杂度
                        if not isinstance(item, dict):
                            continue
                        imgs = item.get("images") or item.get("image_urls") or []
                        if isinstance(imgs, list):
                            for v in imgs:
                                if isinstance(v, str):
                                    extracted.append(v)
                                elif isinstance(v, dict):
                                    u = v.get("url") or v.get("image_url") or v.get("link")
                                    if isinstance(u, str):
                                        extracted.append(u)
                        # 从文本字段正则提取
                        for field in ("content", "snippet", "url", "title", "description"):
                            val = item.get(field)
                            if isinstance(val, str):
                                extracted.extend(img_url_regex.findall(val))

                # 3) 去重并裁剪（前5~10张）
                dedup: List[str] = []
                seen = set()
                for u in extracted:
                    us = u.strip()
                    if us and us not in seen:
                        seen.add(us)
                        dedup.append(us)
                image_urls = dedup[:10]

                response["image_urls"] = image_urls
                logger.info("Tavily web_search: 提取到图片URL数量=%d", len(image_urls))
                if not image_urls:
                    logger.warning("Tavily web_search: 未提取到图片URL，建议调整 query（例如追加 ‘高清 图片’）")

            return response
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise RuntimeError(f"Tavily search failed: {e}")

    def _run(self, *args, **kwargs):
        """同步执行方法未实现，强制使用异步版本"""
        raise NotImplementedError("Use async version only.")


class TavilyExtractTool(BaseTool):
    """Tavily 内容提取工具"""
    
    name: str = Field(default="tavily_extract", description="工具名称")
    description: str = Field(
        default="Extract clean, structured content from one or more URLs.",
        description="工具描述"
    )
    # --- ✅ 移除错误的 client 字段定义 ---
    # client: AsyncTavilyClient = Field(..., description="Tavily 异步客户端实例") # ❌ 不当的定义

    # --- ✅ 新增/修改：显式定义 __init__ 方法 ---
    def __init__(self, client: AsyncTavilyClient):
        """
        初始化 TavilyExtractTool。
        
        Args:
            client (AsyncTavilyClient): 已初始化的 Tavily 异步客户端实例。
        """
        # --- ✅ 关键修改 1: 设置 args_schema ---
        self.args_schema = TavilyExtractParams # ✅ 正确方式
        # --- ✅ 关键修改 2: 存储 client 实例 ---
        self.client = client # ✅ 正确方式
        # --- ✅ 关键修改 3: 调用父类 __init__ ---
        super().__init__( # type: ignore
            name="tavily_extract",
            description="Extract clean, structured content from one or more URLs.",
            args_schema=TavilyExtractParams, # 传入 args_schema 类
            client=client # 传入 client 实例
        )
        # --- ✅ 修改结束 ---

    # --- ✅ 保留 _arun 方法 ---
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        """
        异步执行 Tavily 内容提取。

        Args:
            **kwargs: 提取参数，应符合 TavilyExtractParams 模型。
                      例如: urls=["http://example.com/page1", "http://example.com/page2"]

        Returns:
            Dict[str, Any]: Tavily API 返回的提取结果字典。
            
        Raises:
            ValueError: 如果参数验证失败或 URL 列表为空。
            RuntimeError: 如果 Tavily API 调用失败。
        """
        # --- ✅ 使用 Pydantic 模型验证和解析参数 ---
        try:
            params = TavilyExtractParams(**kwargs)
        except Exception as e:
            logger.error(f"TavilyExtractTool 参数验证失败: {e}")
            raise ValueError(f"参数验证失败: {e}")

        if not params.urls:
            logger.warning("TavilyExtractTool 接收到空的 URL 列表")
            raise ValueError("URL list cannot be empty.")
        try:
            logger.debug(f"调用 Tavily API extract: {params.urls}")
            response = await self.client.extract(urls=params.urls) # type: ignore
            logger.debug(f"Tavily API extract 响应: {response}")
            return response
        except Exception as e:
            logger.error(f"Tavily extract failed: {e}")
            raise RuntimeError(f"Tavily extract failed: {e}")

    def _run(self, *args, **kwargs):
        """同步执行方法未实现，强制使用异步版本"""
        raise NotImplementedError("Use async version only.")


class TavilySearchToolManager:
    """Tavily Search 工具管理器"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 TavilySearchToolManager。

        Args:
            api_key (Optional[str]): Tavily API 密钥。如果未提供，将尝试从环境变量 TAVILY_API_KEY 获取。
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("未提供 Tavily API 密钥，部分功能可能受限。")
        self.client = AsyncTavilyClient(api_key=self.api_key)
        self.tools: List[BaseTool] = []
        self._initialized = False

    async def initialize(self) -> List[BaseTool]:
        """
        异步初始化 Tavily 工具（创建 SDK 客户端并加载工具）。

        Returns:
            List[BaseTool]: 初始化后的工具列表。
        """
        if self._initialized:
            logger.info("Tavily 工具已初始化，返回缓存的工具列表。")
            return self.tools

        logger.info("🔄 正在初始化 TavilySearchToolManager...")
        try:
            # --- ✅ 正确实例化工具 ---
            # 直接将 self.client 作为关键字参数传递给工具的构造函数
            self.tools = [
                TavilySearchTool(client=self.client), # type: ignore # Pydantic 会处理
                TavilyExtractTool(client=self.client), # type: ignore # Pydantic 会处理
            ]
            self._initialized = True
            logger.info(f"✅ Tavily Search 工具加载成功，共 {len(self.tools)} 个")
            return self.tools
        except Exception as e:
            logger.error(f"❌ Tavily 工具初始化失败: {e}")
            # 优雅降级：返回空列表
            self.tools = []
            self._initialized = True # 标记为已尝试初始化
            return self.tools

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        根据工具名称获取工具实例。

        Args:
            name (str): 工具名称。

        Returns:
            Optional[BaseTool]: 找到的工具实例，如果未找到则返回 None。
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    # --- 以下为便捷方法，供内部或外部直接调用 ---

    async def search_and_summarize(self, query: str, days: int = 30) -> str:
        """
        便捷方法：执行搜索并返回 AI 摘要。

        Args:
            query (str): 搜索查询。
            days (int, optional): 限制搜索结果的天数。 Defaults to 30.

        Returns:
            str: AI 生成的摘要。
        """
        if not self._initialized:
            await self.initialize()
            
        search_tool = self.get_tool_by_name("tavily_search")
        if not search_tool:
            raise ValueError("tavily_search 工具未加载")

        try:
            result = await search_tool._arun(query=query, days=days, include_answer=True)
            answer = result.get("answer", "").strip()
            if not answer:
                answer = "未能生成摘要，但可参考以下搜索结果。"
            return answer
        except Exception as e:
            logger.error(f"search_and_summarize 失败: {e}")
            return f"搜索摘要失败: {e}"

    async def search_and_extract(self, query: str, max_urls: int = 3) -> List[Dict[str, str]]:
        """
        便捷方法：先搜索，再提取前 N 个结果的内容。

        Args:
            query (str): 搜索查询。
            max_urls (int, optional): 要提取内容的最大 URL 数量。 Defaults to 3.

        Returns:
            List[Dict[str, str]]: 提取的内容列表，每个元素包含 url, title, raw_content。
        """
        if not self._initialized:
            await self.initialize()
            
        search_tool = self.get_tool_by_name("tavily_search")
        extract_tool = self.get_tool_by_name("tavily_extract")
        if not search_tool or not extract_tool:
            raise ValueError("所需工具未加载")

        try:
            search_result = await search_tool._arun(
                query=query, max_results=max_urls, include_answer=False
            )
            urls = [r["url"] for r in search_result.get("results", [])[:max_urls] if r.get("url")]
            if not urls:
                return []

            extract_result = await extract_tool._arun(urls=urls)
            extracted = []
            for item in extract_result.get("results", []):
                extracted.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "raw_content": (item.get("raw_content", "")[:500] + "...") if item.get("raw_content") else ""
                })
            return extracted
        except Exception as e:
            logger.error(f"search_and_extract 失败: {e}")
            return []


# --- 全局单例（可选）---
_tavily_manager: Optional[TavilySearchToolManager] = None

async def get_tavily_tools(api_key: Optional[str] = None) -> List[BaseTool]:
    """
    获取全局 Tavily 工具列表的便捷函数。

    Args:
        api_key (Optional[str]): Tavily API 密钥。

    Returns:
        List[BaseTool]: 初始化后的 Tavily 工具列表。
    """
    global _tavily_manager
    if _tavily_manager is None:
        _tavily_manager = TavilySearchToolManager(api_key=api_key)
        await _tavily_manager.initialize()
    return _tavily_manager.tools


async def tavily_search_and_summarize(query: str, days: int = 30) -> str:
    """
    全局便捷函数：执行 Tavily 搜索并返回 AI 摘要。

    Args:
        query (str): 搜索查询。
        days (int, optional): 限制搜索结果的天数。 Defaults to 30.

    Returns:
        str: AI 生成的摘要。
    """
    global _tavily_manager
    if _tavily_manager is None:
        await get_tavily_tools()
    assert _tavily_manager is not None
    return await _tavily_manager.search_and_summarize(query, days=days)
