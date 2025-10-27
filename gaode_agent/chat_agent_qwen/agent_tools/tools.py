# agent_tools/tools.py
from pydantic import BaseModel, SecretStr
import os
import tempfile
import subprocess
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
import httpx
from datetime import datetime
from chat_agent_qwen.agent_mcp.agent_mcp_gaode import MCPClient
import asyncio
from chat_agent_qwen.agent_tools.Tavilysearch_tool import tavily_search_and_summarize 
from tavily import TavilyClient  # 同步客户端
from jinja2 import Template
from chat_agent_qwen.utils.security import SecureFileManager, CodeSecurityChecker, SecurityError
from chat_agent_qwen.agent_memory.memory import MemoryManager
import html

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import asyncio
import json
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# ======================
# Tools 模块
# ======================
class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True

class BaseTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = self.define_parameters()
    
    def define_parameters(self) -> List[ToolParameter]:
        """定义工具参数（子类实现）"""
        return []
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """验证参数是否符合要求"""
        for param in self.parameters:
            if param.required and param.name not in params:
                return False
            if param.name in params:
                try:
                    if param.type == "str":
                        params[param.name] = str(params[param.name])
                    elif param.type == "int":
                        params[param.name] = int(params[param.name])
                    elif param.type == "bool":
                        params[param.name] = bool(params[param.name])
                except ValueError:
                    return False
        return True
    
    def execute(self, params: Dict[str, Any]) -> Any:
        """执行工具（子类实现）"""
        raise NotImplementedError("工具执行方法必须由子类实现")
    
    def format_result(self, raw_result: Any) -> str:
        """将工具的原始返回值格式化为用户友好的字符串
        
        Args:
            raw_result: 工具的原始返回值
            
        Returns:
            格式化后的字符串（用于最终展示给用户）
        """
        # ✅ 如果是字典，使用JSON格式化提高可读性
        if isinstance(raw_result, dict):
            try:
                return json.dumps(raw_result, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(raw_result)
        return str(raw_result)

# --- 高德 MCP 工具 ---
class MCPTool(BaseTool):
    def __init__(self, mcp_client):
        super().__init__(
            name="mcp_tool",
            description="调用高德地图的 MCP 服务，支持路线查询（maps_direction_driving等）、POI搜索（maps_text_search）、地理编码（maps_geo）、天气查询（maps_weather）等功能。"
        )
        self.mcp_client = mcp_client
        self._coord_cache = {}  # ✅ 地址→坐标缓存

    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="tool_name",
                type="str",
                description="高德 MCP 内部工具名，如 maps_direction_driving, maps_text_search, maps_geo, maps_weather 等",
                required=True
            ),
            ToolParameter(
                name="parameters",
                type="dict",
                description="传递给 MCP 工具的具体参数，格式根据 tool_name 变化",
                required=True
            )
        ]
    
    async def _get_coordinates(self, address: str) -> str:
        """获取地址的经纬度坐标（带缓存）
        
        Args:
            address: 中文地址
            
        Returns:
            经纬度字符串 "lng,lat"
        """
        # 检查缓存
        if address in self._coord_cache:
            logger.info(f"✅ 命中地址缓存: {address} → {self._coord_cache[address]}")
            return self._coord_cache[address]
        
        # 调用 maps_geo
        try:
            result = await self.mcp_client.run_tool("maps_geo", {"address": address})
            
            if isinstance(result, dict) and result.get("success") and result.get("location"):
                location = result["location"]
                self._coord_cache[address] = location
                logger.info(f"🗺️ 地理编码成功: {address} → {location}")
                return location
            else:
                logger.warning(f"⚠️ 地理编码失败: {address} → {result}")
                return ""  # 返回空字符串表示失败
        except Exception as e:
            logger.error(f"❌ 地理编码异常: {address} → {e}")
            return ""
    
    async def _ensure_locations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """确保 maps_text_search 的结果中每个 POI 都有 location。"""
        if not isinstance(result, dict) or "pois" not in result:
            return result

        pois = result.get("pois", [])
        if not isinstance(pois, list):
            return result

        updated_pois = []
        tasks = []

        async def get_location_for_poi(poi):
            if isinstance(poi, dict) and (not poi.get("location") or poi.get("location") == ""):
                address_to_search = poi.get("address") or poi.get("name")
                if address_to_search:
                    logger.info(f"🔄 POI '{poi.get('name')}' 缺少坐标，正在通过 maps_geo 查询...")
                    coord = await self._get_coordinates(address_to_search)
                    if coord:
                        poi["location"] = coord
                        logger.info(f"✅ 成功获取坐标: {coord}")
                    else:
                        logger.warning(f"⚠️ 未能为 '{address_to_search}' 获取坐标。")
            return poi

        for poi in pois:
            tasks.append(get_location_for_poi(poi))
        
        updated_pois = await asyncio.gather(*tasks)
        result["pois"] = updated_pois
        return result

    def _is_coordinate(self, value: str) -> bool:
        """判断是否为经纬度坐标格式
        
        Args:
            value: 待判断字符串
            
        Returns:
            True 如果是 "lng,lat" 格式
        """
        if not isinstance(value, str):
            return False
        parts = value.split(",")
        if len(parts) != 2:
            return False
        try:
            float(parts[0])
            float(parts[1])
            return True
        except:
            return False
    
    async def _preprocess_parameters(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """✅ 智能预处理参数：自动将中文地址转为坐标
        
        处理规则:
        1. 路线规划工具 (direction_*): origin/destination 必须是坐标
        2. 周边搜索 (around_search): location 必须是坐标
        3. 如果检测到中文地址，自动调用 maps_geo 转换
        
        Args:
            tool_name: MCP工具名
            params: 原始参数
            
        Returns:
            预处理后的参数
        """
        processed = params.copy()
        
        # ✅ 处理路线规划工具
        if "direction" in tool_name:  # maps_direction_driving, maps_direction_walking, maps_direction_bicycling
            for key in ["origin", "destination"]:
                if key in processed:
                    value = str(processed[key])
                    # 检测是否为坐标格式
                    if not self._is_coordinate(value):
                        logger.info(f"🔄 检测到中文地址 '{key}': {value}，正在转换为坐标...")
                        coord = await self._get_coordinates(value)
                        if coord:
                            processed[key] = coord
                            logger.info(f"✅ 已转换 {key}: {value} → {coord}")
                        else:
                            logger.error(f"❌ 无法转换地址 '{value}' 为坐标，路线规划可能失败")
        
        # ✅ 处理周边搜索
        elif tool_name == "maps_around_search":
            if "location" in processed:
                value = str(processed["location"])
                if not self._is_coordinate(value):
                    logger.info(f"🔄 检测到中心点地址: {value}，正在转换...")
                    coord = await self._get_coordinates(value)
                    if coord:
                        processed["location"] = coord
                        logger.info(f"✅ 已转换 location: {value} → {coord}")
        
        return processed

    async def arun(self, params: Dict[str, Any]) -> Any:
        """异步执行方法，现在调用自定义的 MCPClient。"""
        if "tool_name" not in params:
            return {"success": False, "error": "工具调用失败：缺少 'tool_name'"}
        if "parameters" not in params:
            return {"success": False, "error": "工具调用失败：缺少 'parameters'"}
        
        tool_name = params["tool_name"]
        original_params = params["parameters"]
        
        try:
            # ✅ 步骤 1: 获取所有可用的工具方法
            mcp_tool_methods = await self.mcp_client.get_tools()
            if not mcp_tool_methods:
                return {"success": False, "error": "MCP 客户端初始化失败或未能加载任何工具。"}

            # ✅ 步骤 2: 查找目标工具方法
            target_method = mcp_tool_methods.get(tool_name)
            if not target_method:
                return {"success": False, "error": f"工具 '{tool_name}' 未找到。"}

            # ✅ 步骤 3: 预处理参数
            processed_params = await self._preprocess_parameters(tool_name, original_params)
            
            # ✅ 步骤 4: 直接调用找到的方法
            logger.info(f"✅ 找到工具方法 '{tool_name}', 正在调用 with params: {processed_params}")
            result = await target_method(**processed_params)
            
            # ✅ 步骤 5: 后处理结果
            if tool_name == "maps_text_search":
                result = await self._ensure_locations(result)

            if isinstance(result, dict):
                result = self._normalize_result(tool_name, result)
            
            if isinstance(result, dict):
                for key in ["results", "forecasts", "pois", "paths"]:
                    if key in result and isinstance(result[key], list):
                        logger.info(f"🛠️ 工具 {tool_name} 返回 {key}: 长度={len(result[key])}")
            
            return result
        
        except Exception as e:
            error_msg = f"工具调用时发生意外错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
    
    def _normalize_result(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """✅ 标准化 MCP 结果：将 location 字符串转为对象
        
        Args:
            tool_name: 工具名
            result: 原始结果
            
        Returns:
            标准化后的结果
        """
        # ✅ 处理 maps_geo 结果
        if tool_name == "maps_geo" and result.get("success") and result.get("location"):
            loc_str = result["location"]
            if "," in loc_str:
                lng, lat = loc_str.split(",")
                result["location_obj"] = {
                    "lng": float(lng),
                    "lat": float(lat),
                    "location": loc_str  # 保留原始字符串
                }
        
        # ✅ 处理路线规划结果（提取关键字段）
        if "direction" in tool_name and "paths" in result:
            if len(result.get("paths", [])) > 0:
                path = result["paths"][0]
                result["distance_m"] = int(path.get("distance", 0))
                result["duration_s"] = int(path.get("duration", 0))
                result["distance_km"] = round(result["distance_m"] / 1000, 1)
                result["duration_min"] = round(result["duration_s"] / 60, 0)
        
        return result

    def execute(self, params: Dict[str, Any]) -> Any:
        """同步执行方法，内部不推荐使用，提示使用异步调用"""
        raise RuntimeError("MCPTool 应该使用异步方式调用，例如通过 `arun` 或 LangChain 的 `StructuredTool.coroutine`。")

    def format_result(self, raw_result: Any) -> str:
        """格式化高德地图 MCP 工具的返回结果"""
        if isinstance(raw_result, dict):
            # 错误返回
            if "error" in raw_result:
                return f"地图服务错误: {raw_result['error'].get('message', '未知错误')}"

            # 地理编码 maps_geo
            if "results" in raw_result and isinstance(raw_result["results"], list) and len(raw_result["results"]) > 0:
                res = raw_result["results"][0]
                location = res.get("location", "未知")
                address_info = f"{res.get('province','')}{res.get('city','')}{res.get('district','')}"
                return f"地址信息: {address_info}\n经纬度: {location}"

            # 路线规划 maps_direction_driving
            if "routes" in raw_result and isinstance(raw_result["routes"], list) and len(raw_result["routes"]) > 0:
                route = raw_result["routes"][0]
                distance = route.get("distance", 0)
                duration = route.get("duration", 0)
                steps = route.get("paths", [{}])[0].get("steps", [])
                steps_text = "\n".join([f"- {step.get('instruction', '')} ({step.get('distance', 0)}米)" for step in steps[:3]])
                return f"从起点到终点，总距离约 {distance} 米，预计耗时 {duration} 秒。\n详细路线:\n{steps_text}\n(更多步骤请查看完整地图)"

            # 默认返回 JSON 字符串
            return json.dumps(raw_result, ensure_ascii=False)
        
        else:
            # 不是字典，直接返回字符串
            return str(raw_result)
# --- Tavily 网络搜索工具管理器 ---
class TavilySearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="使用 Tavily 实时搜索引擎查询最新网络信息，并返回 AI 生成的摘要和关键结果。适用于查询景点信息、开放时间、门票价格、新闻事件、技术趋势等。输入应为自然语言问题，例如“2025年人工智能发展趋势有哪些？”"
        )
        self.client = TavilyClient(api_key="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")

    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="str",
                description="要搜索的自然语言问题，例如“2025年云南旅游推荐”",
                required=True
            )
        ]

    async def arun(self, **params: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行方法（推荐使用）"""
        query = params.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return {"error": "web_search 工具需要提供有效的 'query' 参数"}

        try:
            response = self.client.search(
                query=query.strip(),
                search_depth="advanced",
                include_answer=True,
                include_images=True,
                max_results=5,
                days=30
            )
            # ✅ 增强：尽可能提取图片 URL（兼容多种响应结构）
            if isinstance(response, dict):
                images_top = response.get("images", [])
                results = response.get("results", []) or []

                extracted: List[str] = []

                # 1) 顶层 images：可能是字符串 URL 列表，或对象列表
                if isinstance(images_top, list):
                    for img in images_top:
                        if isinstance(img, str):
                            extracted.append(img)
                        elif isinstance(img, dict):
                            url = img.get("url") or img.get("image_url") or img.get("link")
                            if isinstance(url, str):
                                extracted.append(url)

                # 2) results[*].images / image_urls / 以及内容中的图片链接
                img_url_regex = re.compile(r'https?://[^\s\'"<>]+?\.(?:jpg|jpeg|png|gif|webp)\b', re.IGNORECASE)
                if isinstance(results, list):
                    for item in results[:10]:  # 限前10条以控复杂度
                        if not isinstance(item, dict):
                            continue
                        imgs = item.get("images") or item.get("image_urls") or []
                        if isinstance(imgs, list):
                            for img in imgs:
                                if isinstance(img, str):
                                    extracted.append(img)
                                elif isinstance(img, dict):
                                    url = img.get("url") or img.get("image_url") or img.get("link")
                                    if isinstance(url, str):
                                        extracted.append(url)
                        # 从文本字段中正则提取潜在图片 URL
                        for field in ("content", "snippet", "url", "title", "description"):
                            val = item.get(field)
                            if isinstance(val, str):
                                extracted.extend(img_url_regex.findall(val))

                # 3) 去重与裁剪
                dedup: List[str] = []
                seen = set()
                for u in extracted:
                    us = u.strip()
                    if us and us not in seen:
                        seen.add(us)
                        dedup.append(us)

                image_urls = dedup[:10]

                logger.info(f"Tavily web_search: 提取到图片URL数量={len(image_urls)} for query='{query[:40]}...' ")
                if len(image_urls) == 0:
                    logger.warning("Tavily web_search: 未从响应中提取到图片URL，建议调整查询词（例如加上 '高清 图片'）")

                return {
                    "answer": (response.get("answer", "") or "").strip(),
                    "results": results[:5],
                    "images": images_top[:5] if isinstance(images_top, list) else [],
                    "image_urls": image_urls
                }
            # 如果不是 dict，直接字符串化返回
            return {"answer": str(response), "image_urls": []}
        except Exception as e:
            return {"error": f"搜索失败: {str(e)}"}

    def execute(self, params: Dict[str, Any]) -> str:
        """同步执行方法（兼容旧代码）"""
        result = asyncio.run(self.arun(**params))
        return self.format_result(result)
    
    def format_result(self, raw_result: Any) -> str:
        if isinstance(raw_result, dict):
            # 如果有错误，直接返回
            if "error" in raw_result:
                return raw_result["error"]
            
            answer = raw_result.get("answer", "").strip()
            image_urls = raw_result.get("image_urls", [])
            
            # 构建格式化输出
            result_parts = []
            if answer:
                result_parts.append(f"📝 搜索摘要：\n{answer}")
            
            if image_urls:
                result_parts.append(f"\n🖼️ 找到 {len(image_urls)} 张相关图片：")
                for i, url in enumerate(image_urls[:3], 1):
                    result_parts.append(f"   {i}. {url}")
            
            return "\n".join(result_parts) if result_parts else "未找到相关搜索结果。"
        
        return str(raw_result)
# --- 可视化工具 ---
logger = logging.getLogger(__name__)


class VisualizationTool(BaseTool):
    def __init__(self, llm_model, memory_manager: MemoryManager):
        super().__init__(
            name="visualization_tool",
            description="生成交互式地图（HTML）和数据图表（Matplotlib）"
        )
        self.memory_manager = memory_manager
        self.llm_model = llm_model

        # 获取高德 API Key（建议从 MCPClient 或环境变量读取）
        self.gaode_api_key = os.getenv("GAODE_API_KEY") or MCPClient().api_key
        if not self.gaode_api_key:
            raise ValueError("GAODE_API_KEY 未设置，请在环境变量或 MCPClient 中配置")

    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="type", type="str", description="可视化类型，'map' 或 'chart'", required=True),
            ToolParameter(name="data", type="dict", description="用于生成可视化的结构化数据", required=True),
            ToolParameter(name="user_id", type="str", description="执行操作的用户ID", required=True)
        ]

    async def arun(self, **params: Dict[str, Any]) -> str:
        """异步执行方法（主要调用接口）"""
        return self.execute(params)
    
    def run(self, **params: Dict[str, Any]) -> str:
        """同步执行方法（使用 asyncio.run 包装 arun）"""
        return asyncio.run(self.arun(**params))
    
    def execute(self, params: Dict[str, Any]) -> str:
        vis_type = params.get("type")
        data = params.get("data")
        # 兜底: 若未显式传入, 从环境变量读取或使用 'anonymous'
        user_id = params.get("user_id") or os.getenv("CURRENT_USER_ID") or ""

        if not user_id:
            return "错误: 'user_id' 是必填参数。建议：请在调用 visualization_tool 时提供 user_id，或确保已设置环境变量 CURRENT_USER_ID。"

        try:
            if vis_type == "map":
                return self._generate_map_html(data, user_id) # type: ignore
            elif vis_type == "chart":
                return self._generate_chart_image(data, user_id) # type: ignore
            else:
                return f"未知的可视化类型: {vis_type}。请使用 'map' 或 'chart'。"
        except Exception as e:
            logger.error(f"可视化工具执行失败: {str(e)}")
            return f"生成可视化失败: {str(e)}"

    def _generate_map_html(self, data: Dict[str, Any], user_id: str) -> str:
        """使用 Jinja2 模板生成高德地图 HTML"""
        # 验证必要字段
        required_keys = ["markers"]
        if not all(k in data for k in required_keys):
            return "地图数据缺少必要字段（如 markers）"
        
        # ✅ 增强：处理 POI 列表的情况，自动提取 location 并转换为 lng/lat，支持 image_urls 多图
        markers_input = data.get("markers", [])
        if markers_input and isinstance(markers_input, list):
            processed_markers = []
            for item in markers_input:
                if isinstance(item, dict):
                    # 统一处理 image_urls（支持多图）
                    image_urls = item.get("image_urls")
                    if isinstance(image_urls, list):
                        # 只保留有效 http/https 链接，最多 3 张
                        image_urls = [u.strip() for u in image_urls if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))][:3]
                    else:
                        image_urls = []
                    # 兼容单图 image_url
                    img = item.get("image_url", "")
                    if isinstance(img, str) and img.strip().lower().startswith(("http://", "https://")):
                        image_url = img.strip()
                    else:
                        image_url = ""
                    # 如果已经有 lng 和 lat，直接使用
                    if "lng" in item and "lat" in item:
                        processed_markers.append({
                            "lng": float(item["lng"]),
                            "lat": float(item["lat"]),
                            "title": item.get("title") or item.get("name", "标记点"),
                            "description": item.get("description") or item.get("address", ""),
                            "image_url": image_url,
                            "image_urls": image_urls
                        })
                    # 如果有 location 字段（"lng,lat" 格式）
                    elif "location" in item and isinstance(item.get("location"), str):
                        try:
                            parts = item["location"].split(",")
                            if len(parts) == 2:
                                processed_markers.append({
                                    "lng": float(parts[0].strip()),
                                    "lat": float(parts[1].strip()),
                                    "title": item.get("name") or item.get("title", "标记点"),
                                    "description": item.get("address") or item.get("description", ""),
                                    "image_url": image_url,
                                    "image_urls": image_urls
                                })
                                logger.info(f"✅ 从 POI 提取坐标: {item.get('name', 'unknown')} -> {item['location']}")
                        except Exception as e:
                            logger.warning(f"⚠️ 无法从 location 提取坐标: {item.get('location')}, 错误: {e}")
                    else:
                        processed_markers.append({
                            "lng": item.get("lng"),
                            "lat": item.get("lat"),
                            "title": item.get("title") or item.get("name", "标记点"),
                            "description": item.get("description") or item.get("address", ""),
                            "image_url": image_url,
                            "image_urls": image_urls
                        })
            markers_input = processed_markers if processed_markers else markers_input

        # ✅ 修复: 预处理和验证标记点坐标有效性
        valid_markers = []
        for marker in markers_input:
            lng = marker.get("lng")
            lat = marker.get("lat")
            
            # ✅ 新增: 处理 "lng,lat" 字符串格式
            if lng is not None and isinstance(lng, str) and ',' in lng and (lat is None or lat == ""):
                try:
                    parts = lng.split(',')
                    if len(parts) == 2:
                        lng = float(parts[0].strip())
                        lat = float(parts[1].strip())
                        logger.info(f"✅ 解析 'lng,lat' 字符串: {marker.get('lng')} → lng={lng}, lat={lat}")
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ 无法解析 'lng,lat' 字符串: {marker.get('lng')}, 错误: {e}")
                    continue
            
            # 检查坐标是否为有效数字
            try:
                lng_float = float(lng) if lng not in [None, ""] else None
                lat_float = float(lat) if lat not in [None, ""] else None
                
                if lng_float is not None and lat_float is not None:
                    # 检查是否在中国范围内 (73-135E, 18-54N)
                    if 73 <= lng_float <= 135 and 18 <= lat_float <= 54:
                        valid_markers.append({
                            "lng": lng_float,
                            "lat": lat_float,
                            "title": marker.get("title", "标记点"),
                            "description": marker.get("description", ""),
                            "image_url": marker.get("image_url", ""),
                            "image_urls": marker.get("image_urls", [])
                        })
                    else:
                        logger.warning(f"⚠️ 跳过无效坐标（超出中国范围）: lng={lng_float}, lat={lat_float}")
                else:
                    logger.warning(f"⚠️ 跳过空坐标: marker={marker.get('title', 'unknown')}")
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ 跳过无效坐标格式: lng={lng}, lat={lat}, error={e}")
        
        if not valid_markers:
            logger.error("❌ 没有有效的标记点，无法生成地图")
            return "地图生成失败：没有有效的坐标数据（请检查 markers 中的 lng 和 lat 字段）"

        # 计算中心点（取第一个有效 marker）
        center = valid_markers[0]

        # ✅ 修复与增强: 构建路线点 polyline_points
        valid_polyline_points = []
        raw_poly_points = data.get("polyline_points", [])

        # 1) 先尝试使用传入的 polyline_points
        if isinstance(raw_poly_points, list) and raw_poly_points:
            for point in raw_poly_points:
                lng = point.get("lng")
                lat = point.get("lat")
                # 处理 "lng,lat" 合并字符串
                if lng is not None and isinstance(lng, str) and ',' in lng and (lat is None or lat == ""):
                    try:
                        parts = lng.split(',')
                        if len(parts) == 2:
                            lng = float(parts[0].strip())
                            lat = float(parts[1].strip())
                    except (ValueError, IndexError):
                        logger.warning(f"⚠️ 无法解析路线点 'lng,lat': {point.get('lng')}")
                        continue
                try:
                    lng_float = float(lng) if lng not in [None, ""] else None
                    lat_float = float(lat) if lat not in [None, ""] else None
                    if lng_float is not None and lat_float is not None:
                        if 73 <= lng_float <= 135 and 18 <= lat_float <= 54:
                            valid_polyline_points.append({"lng": lng_float, "lat": lat_float})
                        else:
                            logger.warning(f"⚠️ 跳过无效路线点: lng={lng_float}, lat={lat_float}")
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ 跳过无效路线点格式: lng={lng}, lat={lat}")

        # 2) 如果没有显式 polyline_points，尝试从路线 steps.polyline 自动解析
        if not valid_polyline_points:
            # 兼容 data.route.paths 或 data.paths
            paths_source = None
            if isinstance(data.get("route"), dict) and isinstance(data["route"].get("paths"), list):
                paths_source = data["route"]["paths"]
            elif isinstance(data.get("paths"), list):
                paths_source = data["paths"]

            try:
                if paths_source and len(paths_source) > 0:
                    steps = paths_source[0].get("steps") or []
                    for st in steps:
                        poly = st.get("polyline")
                        if isinstance(poly, str) and ";" in poly:
                            pairs = poly.split(";")
                            for pair in pairs:
                                try:
                                    lng_str, lat_str = pair.split(",")
                                    lng_f = float(lng_str)
                                    lat_f = float(lat_str)
                                    if 73 <= lng_f <= 135 and 18 <= lat_f <= 54:
                                        valid_polyline_points.append({"lng": lng_f, "lat": lat_f})
                                except Exception:
                                    continue
            except Exception as e:
                logger.warning(f"⚠️ 解析路线 steps.polyline 失败: {e}")

        # 渲染模板
        # 在渲染前清理文本字段，防止模板语法注入
        for m in valid_markers:
            for k in ["title", "description"]:
                if k in m and isinstance(m[k], str):
                    # 删除 Jinja2 模板符号，替换为安全字符
                    cleaned = m[k].replace('{{', '').replace('}}', '')
                    # HTML 转义，防止 XSS
                    m[k] = html.escape(cleaned)

        # 提取路线摘要和步骤
        route_summary = data.get("route_summary", {})
        route_steps = data.get("route_steps", [])

        # 如果摘要或步骤为空，但有原始路径数据，则尝试从中提取
        if (not route_summary or not route_steps) and data.get("paths"):
            path = data["paths"][0] if data.get("paths") else {}
            route_summary = {
                "distance": path.get("distance", 0),
                "duration": path.get("duration", 0)
            }
            route_steps = path.get("steps", [])

        # 使用自动解析的 polyline_points（若存在）
        if valid_polyline_points:
            data["polyline_points"] = valid_polyline_points


        html_content = self._render_map_template(
            title=html.escape(str(data.get("title", "路线地图"))),
            center=center,
            markers=valid_markers,  # ✅ 使用验证后的标记
            polyline_points=valid_polyline_points,  # ✅ 使用验证/解析后的路线点
            route_summary=route_summary,
            route_steps=route_steps
        )
        
        # ✅ 使用 MemoryManager 获取文件路径
        absolute_path = self.memory_manager.add_file_reference(
            user_id=user_id,
            file_type="html",
            file_name=f"map_{uuid.uuid4().hex[:8]}.html",
            description=data.get("title", "交互式地图")
        )
        
        # 生成用户可见路径（相对路径）
        import hashlib
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()
        user_facing_path = str(absolute_path.relative_to(Path(".")))

        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # ✅ 新功能: 自动用浏览器打开 HTML 文件（避免被 Word 等关联程序接管）
        try:
            import webbrowser
            import subprocess
            import platform
            abs_path = Path(absolute_path).resolve()
            abs_uri = abs_path.as_uri()  # file:// URI
            opened = False
            # 首选使用 webbrowser（通常会调用默认浏览器）
            try:
                opened = bool(webbrowser.open(abs_uri, new=2))
                if opened:
                    logger.info(f"🌐 已通过 webbrowser 打开: {abs_uri}")
            except Exception as e1:
                logger.warning(f"⚠️ webbrowser.open 失败: {e1}")
            # Windows 强制用 Edge 作为兜底，避免 .html 关联到 Word
            if not opened and platform.system() == 'Windows':
                try:
                    # 使用 microsoft-edge: 协议强制调用 Edge 浏览器
                    cmd = ["cmd", "/c", "start", "", f"microsoft-edge:{abs_uri}"]
                    subprocess.run(cmd, shell=False)
                    opened = True
                    logger.info("🌐 已通过 Microsoft Edge 强制打开 HTML")
                except Exception as e2:
                    logger.warning(f"⚠️ 打开 Edge 失败: {e2}")
            # macOS 兜底: 使用 open 命令
            if not opened and platform.system() == 'Darwin':
                try:
                    subprocess.run(["open", abs_uri])
                    opened = True
                except Exception as e3:
                    logger.warning(f"⚠️ macOS open 失败: {e3}")
            # Linux 兜底: 使用 xdg-open
            if not opened and platform.system() == 'Linux':
                try:
                    subprocess.run(["xdg-open", abs_uri])
                    opened = True
                except Exception as e4:
                    logger.warning(f"⚠️ xdg-open 失败: {e4}")
        except Exception as e:
            logger.warning(f"⚠️ 无法自动打开浏览器: {e}")

        logger.info(f"✅ 地图 HTML 文件已生成: {absolute_path} (返回: {user_facing_path})")
        logger.info(f"✅ 有效标记点: {len(valid_markers)}, 有效路线点: {len(valid_polyline_points)}")
        return user_facing_path

    def _render_map_template(self, title: str, center: dict, markers: list, polyline_points: list, route_summary: dict, route_steps: list) -> str:
        """从文件加载并渲染 Jinja2 地图模板"""
        # ✅ 在try块外初始化template_path
        template_path = Path(__file__).parent / "map_template.html"
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
            
            template = Template(template_str)
            
            return template.render(
                title=title,
                gaode_api_key=self.gaode_api_key,
                center=center,
                markers=markers,
                polyline_points=polyline_points,
                route_summary=route_summary,
                route_steps=route_steps
            )
        except FileNotFoundError:
            logger.error(f"地图模板文件未找到: {template_path}")
            return "错误: 地图模板文件 'map_template.html' 未找到。"
        except Exception as e:
            logger.error(f"渲染地图模板时出错: {e}")
            return f"错误: 渲染地图模板时出错: {e}"

    def _generate_chart_image(self, data: Dict[str, Any], user_id: str) -> str:
        """使用固定模板生成 Matplotlib 图表"""
        # 从 LLM 获取图表配置（仅类型和标题）
        chart_config = self._get_chart_config_from_llm(data)
        if not chart_config:
            return "无法解析图表配置"

        # 验证数据
        x_data = data.get("x_data")
        y_data = data.get("y_data")
        if not x_data or not y_data or len(x_data) != len(y_data):
            return "图表数据格式错误：x_data 和 y_data 必须等长且非空"

        # ✅ 使用 MemoryManager 获取文件路径
        absolute_path = self.memory_manager.add_file_reference(
            user_id=user_id,
            file_type="png",
            file_name=f"chart_{uuid.uuid4().hex[:8]}.png",
            description=chart_config.get("title", "数据图表")
        )
        user_facing_path = str(absolute_path.relative_to(Path(".")))

        # 构建安全绘图脚本
        chart_script = self._build_chart_script(
            chart_type=chart_config["type"],
            title=chart_config["title"],
            x_label=data.get("x_label", "X 轴"),
            y_label=data.get("y_label", "Y 轴"),
            x_data=x_data,
            y_data=y_data,
            output_path=str(absolute_path)
        )

        # 执行脚本（在子进程中）
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(chart_script)
                tmp_path = tmp.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=15
            )
            os.unlink(tmp_path)

            if result.returncode != 0:
                logger.error(f"图表生成失败: {result.stderr}")
                return f"图表生成失败: {result.stderr[:200]}..."
            else:
                logger.info(f"图表 PNG 文件已生成: {absolute_path}")
                return user_facing_path

        except Exception as e:
            logger.error(f"图表生成异常: {e}")
            return f"图表生成异常: {str(e)}"

    def _get_chart_config_from_llm(self, data: Dict[str, Any]) -> Dict[str, str]:
        """让 LLM 仅决定图表类型和标题，不生成代码"""
        prompt = f"""
        你是一个数据可视化专家。请根据以下数据描述，仅输出一个 JSON 对象，包含：
        - "type": 图表类型，必须是 "bar"（柱状图）、"line"（折线图）、"pie"（饼图）之一
        - "title": 图表标题（中文）

        数据描述: {data.get('description', '未提供')}
        示例输出: {{"type": "bar", "title": "2025年旅游推荐"}}
        """
        try:
            response = self.llm_model.generate([{"role": "user", "content": prompt}]).strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0]
            config = json.loads(response)
            if config["type"] not in ["bar", "line", "pie"]:
                config["type"] = "bar"
            return config
        except:
            return {"type": "bar", "title": "数据图表"}

    def _build_chart_script(self, chart_type: str, title: str, x_label: str, y_label: str,
                           x_data: list, y_data: list, output_path: str) -> str:
        """构建安全的 Matplotlib 脚本"""
        script = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

x_data = {repr(x_data)}
y_data = {repr(y_data)}

plt.figure(figsize=(10, 6))
"""
        if chart_type == "bar":
            script += "plt.bar(x_data, y_data)\n"
        elif chart_type == "line":
            script += "plt.plot(x_data, y_data, marker='o')\n"
        elif chart_type == "pie":
            script += "plt.pie(y_data, labels=x_data, autopct='%1.1f%%')\n"
            script += "plt.title({repr(title)})\n"
            script += f"plt.savefig({repr(output_path)}, dpi=300, bbox_inches='tight')\n"
            script += "plt.close()\n"
            return script

        script += f"""
plt.title({repr(title)})
plt.xlabel({repr(x_label)})
plt.ylabel({repr(y_label)})
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig({repr(output_path)}, dpi=300, bbox_inches='tight')
plt.close()
"""
        return script
# --- 文件工具 ---
class FileTool(BaseTool):
    def __init__(self, llm_model, memory_manager: MemoryManager):
        super().__init__(
            name="file_tool",
            description="生成 PDF 和 Excel 格式的行程文件"
        )
        self.memory_manager = memory_manager
        self.llm_model = llm_model # 传入 LLM 模型实例，用于生成代码

    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="format", type="str", description="文件格式，'pdf'、'excel' 或 'html'", required=True),
            ToolParameter(name="data", type="dict", description="用于生成文件的数据，格式根据 format 变化", required=True),
            ToolParameter(name="user_id", type="str", description="执行操作的用户ID", required=True)
        ]

    async def arun(self, **params: Dict[str, Any]) -> str:
        """异步执行方法（主要调用接口）"""
        return self.execute(params)
    
    def run(self, **params: Dict[str, Any]) -> str:
        """同步执行方法（使用 asyncio.run 包装 arun）"""
        return asyncio.run(self.arun(**params))
    
    def execute(self, params: Dict[str, Any]) -> str:
        file_format = params.get("format")
        data = params.get("data")
        # 兜底: 若未显式传入, 从环境变量读取或使用 'anonymous'
        user_id = params.get("user_id") or os.getenv("CURRENT_USER_ID") or ""

        if not user_id:
            return "错误: 'user_id' 是必填参数。建议：请在调用 file_tool 时提供 user_id，或确保已设置环境变量 CURRENT_USER_ID。"

        try:
            if file_format == "pdf" or file_format == "txt":
                logger.warning(f"PDF/TXT 文件生成功能已禁用，收到 format={file_format} 请求，直接返回错误。")
                return f"错误: 当前 PDF/TXT 文件生成功能已禁用，请使用 'excel' 或 'html' 格式。"
            elif file_format == "excel":
                return self._generate_excel(data, user_id) # type: ignore
            elif file_format == "html":  # ✅ 新增: 支持 HTML 格式
                return self._generate_html(data, user_id) # type: ignore
            else:
                return f"未知的文件格式: {file_format}。请使用 'excel' 或 'html'。"
        except Exception as e:
            logger.error(f"文件工具执行失败: {str(e)}")
            return f"生成文件失败: {str(e)}"

    def _generate_pdf(self, data: Dict[str, Any], user_id: str) -> str:
        """生成 PDF 行程文件，通过 LLM 生成代码"""
        # 准备 LLM 生成代码的 Prompt
        prompt = f"""
        你的任务是根据提供的行程数据，生成一段 Python 代码，使用 reportlab 库创建一个格式规范、内容完整的 PDF 文件。
        要求:
        1.  使用 reportlab 库。
        2.  页面设置: A4 纸张，边距合理。
        3.  内容结构:
           - 封面页: 包含行程名称、天数、目的地。
           - 内容页: 按天数分组，使用表格展示每日行程 (时间、景点、交通、耗时、备注)。
           - (可选) 嵌入地图或图表图片 (如果 Agent 提供了图片路径)。
           - 实用提示页: 包含预约链接、注意事项、紧急联系人等。
        4.  格式优化:
           - 使用不同字体大小区分标题和内容。
           - 为表头添加背景色。
           - 适当使用颜色标记重要信息 (如 "需预约")。
           - 布局清晰，阅读友好。
        5.  代码应包含必要的库导入、文档创建、内容添加、保存为文件的逻辑。
        6.  代码结构清晰，注释明确，易于理解。
        7.  输出应为可直接执行的 Python 代码字符串。
        输入数据 (JSON 格式): {json.dumps(data, ensure_ascii=False)}
        请直接输出生成的 Python 代码。
        """

        # 调用 LLM 生成代码
        python_code = self.llm_model.generate([{"role": "user", "content": prompt}]).strip() # type: ignore

        # --- 安全检查：对生成的 Python 代码进行安全检查 ---
        if not self._is_code_safe(python_code):
             logger.error("Generated Python code for PDF is unsafe.")
             return "生成的 PDF 代码不安全，无法执行。"

        # ✅ 使用 MemoryManager 获取文件路径
        absolute_path = self.memory_manager.add_file_reference(
            user_id=user_id,
            file_type="pdf",
            file_name=f"itinerary_{uuid.uuid4().hex[:8]}.pdf",
            description=data.get("title", "行程单")
        )
        user_facing_path = str(absolute_path.relative_to(Path(".")))

        # 在沙箱环境中执行生成的代码
        # **警告：在生产环境中，必须使用安全的沙箱环境执行 LLM 生成的代码**
        try:
            # 将生成的代码写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # 在代码末尾添加保存文件的指令
                code_with_save = python_code + f"\ndoc.build(story)\n" # reportlab 通常以 doc.build 结尾
                # 假设生成的代码中 doc = SimpleDocTemplate(...) 已经指定了 file_path
                # 如果没有，需要在 prompt 中强调
                # 这里假设 LLM 生成的代码已经包含了正确的文件路径
                temp_file.write(code_with_save)
                temp_file_path = temp_file.name

            # 执行临时文件
            result = subprocess.run(
                ["python", temp_file_path],
                capture_output=True,
                text=True,
                timeout=20 # 设置较长超时，生成 PDF 可能较慢
            )

            # 清理临时文件
            os.unlink(temp_file_path)

            if result.returncode != 0:
                logger.error(f"执行 PDF 生成代码失败: {result.stderr}")
                return f"执行 PDF 生成代码失败: {result.stderr.strip()}"
            else:
                # 检查文件是否确实生成
                if absolute_path.exists():
                    logger.info(f"PDF 文件已生成: {absolute_path}")
                    return user_facing_path # 返回文件路径
                else:
                    logger.error(f"PDF 生成代码执行成功，但文件未找到: {absolute_path}")
                    return f"PDF 生成代码执行成功，但文件未找到: {absolute_path}"

        except subprocess.TimeoutExpired:
            logger.error("PDF 生成代码执行超时")
            return "PDF 生成代码执行超时"
        except Exception as e:
            logger.error(f"执行 PDF 生成代码时发生错误: {e}")
            return f"执行 PDF 生成代码时发生错误: {str(e)}"


    def _generate_excel(self, data: Dict[str, Any], user_id: str) -> str:
        """生成 Excel 行程文件，通过 LLM 生成代码"""
        # 准备 LLM 生成代码的 Prompt
        prompt = f"""
        你的任务是根据提供的行程数据，生成一段 Python 代码，使用 openpyxl 库创建一个结构清晰、格式美观的 Excel 文件。
        要求:
        1.  使用 openpyxl 库。
        2.  创建至少一个 "行程表" 工作表。
        3.  表格列: 包含 "日期", "时间段", "景点", "交通方式", "耗时", "费用", "备注" 等。
        4.  格式优化:
           - 设置表头行背景色。
           - 自动调整列宽以适应内容。
           - 为单元格添加边框。
           - (可选) 为特定列 (如费用) 设置数字格式。
        5.  代码应包含必要的库导入、工作簿/工作表创建、数据填充、格式设置、保存为文件的逻辑。
        6.  代码结构清晰，注释明确，易于理解。
        7.  输出应为可直接执行的 Python 代码字符串。
        输入数据 (JSON 格式): {json.dumps(data, ensure_ascii=False)}
        请直接输出生成的 Python 代码。
        """

        # 调用 LLM 生成代码
        python_code = self.llm_model.generate([{"role": "user", "content": prompt}]).strip() # type: ignore

        # --- 安全检查：对生成的 Python 代码进行安全检查 ---
        if not self._is_code_safe(python_code):
             logger.error("Generated Python code for Excel is unsafe.")
             return "生成的 Excel 代码不安全，无法执行。"

        # ✅ 使用 MemoryManager 获取文件路径
        absolute_path = self.memory_manager.add_file_reference(
            user_id=user_id,
            file_type="xlsx",
            file_name=f"itinerary_{uuid.uuid4().hex[:8]}.xlsx",
            description=data.get("title", "行程表")
        )
        user_facing_path = str(absolute_path.relative_to(Path(".")))

        # 在沙箱环境中执行生成的代码
        # **警告：在生产环境中，必须使用安全的沙箱环境执行 LLM 生成的代码**
        try:
            # 将生成的代码写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                # 在代码末尾添加保存文件的指令
                code_with_save = python_code + f"\nwb.save('{absolute_path}')\n" # openpyxl 通常以 wb.save 结尾
                temp_file.write(code_with_save)
                temp_file_path = temp_file.name

            # 执行临时文件
            result = subprocess.run(
                ["python", temp_file_path],
                capture_output=True,
                text=True,
                timeout=15 # 设置超时
            )

            # 清理临时文件
            os.unlink(temp_file_path)

            if result.returncode != 0:
                logger.error(f"执行 Excel 生成代码失败: {result.stderr}")
                return f"执行 Excel 生成代码失败: {result.stderr.strip()}"
            else:
                # 检查文件是否确实生成
                if absolute_path.exists():
                    logger.info(f"Excel 文件已生成: {absolute_path}")
                    return user_facing_path # 返回文件路径
                else:
                    logger.error(f"Excel 生成代码执行成功，但文件未找到: {absolute_path}")
                    return f"Excel 生成代码执行成功，但文件未找到: {absolute_path}"

        except subprocess.TimeoutExpired:
            logger.error("Excel 生成代码执行超时")
            return "Excel 生成代码执行超时"
        except Exception as e:
            logger.error(f"执行 Excel 生成代码时发生错误: {e}")
            return f"执行 Excel 生成代码时发生错误: {str(e)}"

    def _is_code_safe(self, code: str) -> Tuple[bool, str]:
        """增强的代码安全检查
        
        Returns:
            (is_safe, reason)
        """
        # ✅ 使用安全检查器
        is_safe, reason = CodeSecurityChecker.check_code_safety(code)
        if not is_safe:
            return False, reason
        
        # ✅ 验证导入语句
        import_valid, import_reason = CodeSecurityChecker.validate_imports(code)
        if not import_valid:
            return False, import_reason
        
        return True, "代码安全"

    def _generate_html(self, data: Dict[str, Any], user_id: str) -> str:
        """✅ 新增: 生成 HTML 文件
        
        Args:
            data: 包含 'content' 字段的字典，HTML 内容字符串
        
        Returns:
            生成的 HTML 文件路径
        """
        # 支持两种输入形式：
        # 1) data 包含 'content' (HTML 字符串) 和可选 'filename'
        # 2) data 是完整的行程结构（非 HTML） -> 返回明确错误，提示应先渲染为 HTML 字符串
        html_content = data.get("content")
        filename = data.get("filename") or data.get("file_name") or "document.html"

        if not html_content or not isinstance(html_content, str):
            return "错误: HTML 文件生成需要提供 'content' 字段（HTML 字符串）。如果您传入的是行程数据，请先让 Agent 或 LLM 将其渲染为 HTML 字符串后再调用 file_tool。"

        # 安全检查：禁止包含模板语法（例如 Jinja2 的 {{ }} 或 {% %}）
        if '{{' in html_content or '}}' in html_content or '{%' in html_content or '%}' in html_content:
            logger.error("检测到 HTML 内容包含模板语法（如 '{{' 或 '{%'），为安全起见拒绝生成此文件。")
            return "错误: HTML 内容包含不安全的模板语法（例如 '{{' 或 '{%}'），已拒绝生成。请提供纯静态HTML或先在Agent端渲染。"

        # ✅ 使用 MemoryManager 获取文件路径
        absolute_path = self.memory_manager.add_file_reference(
            user_id=user_id,
            file_type="html",
            file_name=filename,
            description=data.get("title", "HTML文档")
        )
        user_facing_path = str(absolute_path.relative_to(Path(".")))

        try:
            # 直接写入 HTML 内容
            with open(absolute_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML 文件已生成: {absolute_path}")
            return user_facing_path
        except Exception as e:
            logger.error(f"生成 HTML 文件失败: {e}")
            return f"生成 HTML 文件失败: {str(e)}"


# --- 其他现有工具 (保持不变) ---
class SecureCodeInterpreterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="secure_python_interpreter",
            description="在安全环境中执行Python代码并返回结果"
        )
        # 安全限制 - 只允许使用这些模块
        self.allowed_modules = ["math", "datetime", "json", "random", "re", "collections", "matplotlib", "reportlab", "openpyxl"]
        
        # 使用SecretStr存储API密钥（如果需要）
        self.api_key = SecretStr("")  # 可以配置API密钥
    
    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="code", type="str", description="要执行的Python代码", required=True),
            ToolParameter(name="timeout", type="int", description="执行超时时间(秒)", required=False)
        ]
    
    def execute(self, params: Dict[str, Any]) -> str:
        try:
            code = params.get("code", "")
            timeout = params.get("timeout", 10)  # 默认10秒超时
            
            # 安全检查
            if not self._is_code_safe(code):
                return "代码包含不安全操作，拒绝执行"
            
            # 在实际实现中，应使用沙箱环境执行代码
            # 这里简化处理，使用子进程执行
            result = self._execute_in_subprocess(code, timeout)
            return result
        except Exception as e:
            logger.error(f"代码执行失败: {str(e)}")
            return f"代码执行错误: {str(e)}"
    
    def _is_code_safe(self, code: str) -> bool:
        """检查代码安全性"""
        # 禁止危险操作
        dangerous_patterns = [
            r"__import__\s*\(", r"open\s*\(", r"os\.", r"subprocess\.",
            r"exec\s*\(", r"eval\s*\(", r"shutil\.", r"sys\.", r"import\s+os",
            r"import\s+sys", r"import\s+subprocess", r"import\s+shutil"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False
        
        # 检查允许的模块
        import_lines = re.findall(r"import\s+([\w\.]+)", code)
        for imp in import_lines:
            if imp.split(".")[0] not in self.allowed_modules:
                return False
        
        return True
    
    def _execute_in_subprocess(self, code: str, timeout: int = 10) -> str:
        """在子进程中执行代码并获取输出"""
        tmp_path = None  # ✅ 初始化变量
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                tmp.write(code.encode('utf-8'))
                tmp_path = tmp.name
            
            # 执行Python文件
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # 清理临时文件
            os.unlink(tmp_path)
            
            if result.returncode == 0:
                return result.stdout.strip() or "代码执行成功，但无输出"
            else:
                return f"执行错误: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "执行超时"
        finally:
            if tmp_path and os.path.exists(tmp_path):  # ✅ 检查 tmp_path 是否已初始化
                os.unlink(tmp_path)

class FileRunnerTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="run_created_file",
            description="执行Agent创建的文件（支持Python脚本）"
        )
        # 存储Agent创建的文件
        self.agent_files = {}
        
        # 安全限制 - 只允许执行Python文件
        self.allowed_extensions = [".py",".html"]
    
    def define_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="file_id", type="str", description="要执行的文件ID", required=True),
            ToolParameter(name="timeout", type="int", description="执行超时时间(秒)", required=False)
        ]
    
    def save_file(self, content: str, extension: str) -> str:
        """保存Agent创建的文件并返回文件ID"""
        # 检查扩展名是否允许
        if extension not in self.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {extension}")
        
        # 生成唯一文件ID
        file_id = f"agent_file_{len(self.agent_files) + 1}"
        
        # 保存文件内容
        self.agent_files[file_id] = {
            "content": content,
            "extension": extension
        }
        
        return file_id
    
    def execute(self, params: Dict[str, Any]) -> str:
        try:
            # 支持两种用法:
            # 1) 通过 file_id 执行已由 save_file 存储的内容
            # 2) 直接传入 file_path（字符串），打开该路径或执行（如果是 .py）
            timeout = params.get("timeout", 30)  # 默认30秒超时

            if "file_path" in params and params.get("file_path"):
                # ✅ 添加类型检查,确保file_path不是None
                file_path_str = params.get("file_path")
                if not file_path_str:
                    return "错误: file_path 参数为空"
                file_path = Path(file_path_str)
                if not file_path.exists():
                    return f"文件未找到: {file_path}。请先生成文件或传入正确的路径。"

                ext = file_path.suffix.lower()
                if ext == ".py":
                    return self._run_python_file(str(file_path), timeout)
                else:
                    # 尝试用系统默认程序打开（更适合 HTML 等）
                    try:
                        if os.name == 'nt':
                            os.startfile(str(file_path))
                        else:
                            import webbrowser
                            webbrowser.open(str(file_path))
                        return f"已打开文件: {file_path}"
                    except Exception as e:
                        logger.error(f"打开文件失败: {e}")
                        return f"打开文件失败: {e}"

            file_id = params.get("file_id", "")
            if not file_id:
                return "错误: 缺少 file_id 或 file_path 参数"

            timeout = params.get("timeout", 30)

            # 获取文件内容
            if file_id not in self.agent_files:
                return f"文件ID '{file_id}' 不存在"

            file_data = self.agent_files[file_id]
            content = file_data["content"]
            extension = file_data["extension"]

            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
                tmp.write(content.encode('utf-8'))
                tmp_path = tmp.name

            # 根据文件类型执行
            output = ""
            if extension == ".py":
                output = self._run_python_file(tmp_path, timeout)
            else:
                # 非 python 文件直接返回路径
                output = tmp_path

            # 清理临时文件仅在我们不需要持久化时
            try:
                if extension == ".py":
                    os.unlink(tmp_path)
            except Exception:
                pass

            return output
        except Exception as e:
            logger.error(f"文件执行失败: {str(e)}")
            return f"文件执行错误: {str(e)}"
    
    def _run_python_file(self, file_path: str, timeout: int) -> str:
        """执行Python文件"""
        try:
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip() or "Python脚本执行成功，但无输出"
            else:
                return f"Python脚本执行错误: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "执行超时"
