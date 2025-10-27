# chat_agent_qwen/agent_mcp/agent_mcp_gaode.py
import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
# --- 新增：直接调用高德 Web API 的客户端 ---
class GaodeWebAPIClient:
    """
    直接使用 httpx 调用高德地图 Web API v3/v5 的客户端。
    支持高德MCP协议的15个标准工具。
    """
    BASE_URL_V3 = "https://restapi.amap.com/v3"
    BASE_URL_V5 = "https://restapi.amap.com/v5"

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """通用异步请求方法"""
        params['key'] = self.api_key
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "1":
                error_info = data.get("info", "未知错误")
                logger.error(f"高德 API 返回错误: {error_info} (infocode: {data.get('infocode')})")
                return {"success": False, "error": f"高德 API 错误: {error_info}"}
            # ✅ 标准化成功响应
            data["success"] = True
            return data
        except httpx.RequestError as e:
            logger.error(f"请求高德 API 时发生网络错误: {e}")
            return {"success": False, "error": f"网络错误: {str(e)}"}
        except Exception as e:
            logger.error(f"处理高德 API 响应时发生未知错误: {e}")
            return {"success": False, "error": f"未知错误: {str(e)}"}

    def _parse_location(self, location_str: str) -> Tuple[float, float]:
        """解析经纬度字符串为浮点数元组"""
        try:
            parts = location_str.split(",")
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
        except:
            pass
        return 0.0, 0.0

    async def maps_text_search(self, keywords: str, city: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        POI 文本搜索 API (v3)
        https://lbs.amap.com/api/webservice/guide/api/search
        """
        url = f"{self.BASE_URL_V3}/place/text"
        params = {"keywords": keywords, **kwargs}
        if city:
            params["city"] = city
        result = await self._request(url, params)
        
        # ✅ 增强：为每个POI添加解析后的lng/lat字段
        if result.get("success") and "pois" in result:
            for poi in result.get("pois", []):
                if "location" in poi and isinstance(poi["location"], str):
                    lng, lat = self._parse_location(poi["location"])
                    poi["lng"] = lng
                    poi["lat"] = lat
                    logger.debug(f"✅ POI '{poi.get('name')}' 坐标解析: {poi['location']} → lng={lng}, lat={lat}")
        return result

    async def geocode_geo(self, address: str, city: Optional[str] = None) -> Dict[str, Any]:
        """
        地理编码 API (v3) - 将地址转换为坐标
        https://lbs.amap.com/api/webservice/guide/api/georegeo
        """
        url = f"{self.BASE_URL_V3}/geocode/geo"
        params = {"address": address}
        if city:
            params["city"] = city
        result = await self._request(url, params)
        
        # ✅ 增强：为geocode结果添加解析后的lng/lat字段
        if result.get("success") and "geocodes" in result:
            for geocode in result.get("geocodes", []):
                if "location" in geocode and isinstance(geocode["location"], str):
                    lng, lat = self._parse_location(geocode["location"])
                    geocode["lng"] = lng
                    geocode["lat"] = lat
                    logger.debug(f"✅ 地址 '{address}' 坐标解析: {geocode['location']} → lng={lng}, lat={lat}")
        return result
    
    async def maps_direction_driving(self, origin: str, destination: str, **kwargs) -> Dict[str, Any]:
        """驾车路线规划 API (v3)"""
        url = f"{self.BASE_URL_V3}/direction/driving"
        params = {"origin": origin, "destination": destination, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：提取关键路线信息
        if result.get("success") and "route" in result:
            route = result["route"]
            if "paths" in route and len(route["paths"]) > 0:
                path = route["paths"][0]
                result["distance_m"] = int(path.get("distance", 0))
                result["duration_s"] = int(path.get("duration", 0))
                result["distance_km"] = round(result["distance_m"] / 1000, 1)
                result["duration_min"] = round(result["duration_s"] / 60, 0)
        return result
    
    async def maps_direction_walking(self, origin: str, destination: str, **kwargs) -> Dict[str, Any]:
        """步行路线规划 API (v3)"""
        url = f"{self.BASE_URL_V3}/direction/walking"
        params = {"origin": origin, "destination": destination, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：提取关键路线信息
        if result.get("success") and "route" in result:
            route = result["route"]
            if "paths" in route and len(route["paths"]) > 0:
                path = route["paths"][0]
                result["distance_m"] = int(path.get("distance", 0))
                result["duration_s"] = int(path.get("duration", 0))
                result["distance_km"] = round(result["distance_m"] / 1000, 1)
                result["duration_min"] = round(result["duration_s"] / 60, 0)
        return result
    
    async def maps_direction_bicycling(self, origin: str, destination: str, **kwargs) -> Dict[str, Any]:
        """骑行路线规划 API (v4)"""
        url = f"{self.BASE_URL_V3}/direction/bicycling"
        params = {"origin": origin, "destination": destination, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：提取关键路线信息
        if result.get("success") and "route" in result:
            route = result["route"]
            if "paths" in route and len(route["paths"]) > 0:
                path = route["paths"][0]
                result["distance_m"] = int(path.get("distance", 0))
                result["duration_s"] = int(path.get("duration", 0))
                result["distance_km"] = round(result["distance_m"] / 1000, 1)
                result["duration_min"] = round(result["duration_s"] / 60, 0)
        return result
    
    async def maps_direction_transit_integrated(self, origin: str, destination: str, city: str, **kwargs) -> Dict[str, Any]:
        """公交路线规划 API (v3)"""
        url = f"{self.BASE_URL_V3}/direction/transit/integrated"
        params = {"origin": origin, "destination": destination, "city": city, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：提取关键路线信息（公交路线结构略有不同）
        if result.get("success") and "route" in result:
            route = result["route"]
            if "transits" in route and len(route["transits"]) > 0:
                transit = route["transits"][0]
                result["distance_m"] = int(transit.get("distance", 0))
                result["duration_s"] = int(transit.get("duration", 0))
                result["distance_km"] = round(result["distance_m"] / 1000, 1)
                result["duration_min"] = round(result["duration_s"] / 60, 0)
        return result
    
    async def maps_around_search(self, keywords: str, location: str, radius: int = 1000, **kwargs) -> Dict[str, Any]:
        """周边搜索 API (v3)"""
        url = f"{self.BASE_URL_V3}/place/around"
        params = {"keywords": keywords, "location": location, "radius": radius, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：为POI添加坐标解析
        if result.get("success") and "pois" in result:
            for poi in result.get("pois", []):
                if "location" in poi and isinstance(poi["location"], str):
                    lng, lat = self._parse_location(poi["location"])
                    poi["lng"] = lng
                    poi["lat"] = lat
        return result
    
    async def maps_weather(self, city: str, **kwargs) -> Dict[str, Any]:
        """天气查询 API (v3)"""
        url = f"{self.BASE_URL_V3}/weather/weatherInfo"
        params = {"city": city, "extensions": "all", **kwargs}
        return await self._request(url, params)
    
    async def maps_regeocode(self, location: str, **kwargs) -> Dict[str, Any]:
        """逆地理编码 API (v3) - 坐标转地址"""
        url = f"{self.BASE_URL_V3}/geocode/regeo"
        params = {"location": location, **kwargs}
        return await self._request(url, params)
    
    async def maps_ip_location(self, ip: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """IP定位 API (v3)"""
        url = f"{self.BASE_URL_V3}/ip"
        params = {**kwargs}
        if ip:
            params["ip"] = ip
        return await self._request(url, params)
    
    async def maps_distance(self, origins: str, destination: str, **kwargs) -> Dict[str, Any]:
        """距离测量 API (v3)"""
        url = f"{self.BASE_URL_V3}/distance"
        params = {"origins": origins, "destination": destination, **kwargs}
        return await self._request(url, params)
    
    async def maps_search_detail(self, id: str, **kwargs) -> Dict[str, Any]:
        """POI详情查询 API (v3)"""
        url = f"{self.BASE_URL_V3}/place/detail"
        params = {"id": id, **kwargs}
        result = await self._request(url, params)
        
        # ✅ 增强：为POI详情添加坐标解析
        if result.get("success") and "pois" in result:
            for poi in result.get("pois", []):
                if "location" in poi and isinstance(poi["location"], str):
                    lng, lat = self._parse_location(poi["location"])
                    poi["lng"] = lng
                    poi["lat"] = lat
                    logger.debug(f"✅ POI详情 '{poi.get('name')}' 坐标解析: {poi['location']} → lng={lng}, lat={lat}")
        return result

# --- 重构：MCPClient 现在使用我们自己的 GaodeWebAPIClient ---
class MCPClient:
    """
    封装 GaodeWebAPIClient，为上层提供统一的工具调用接口。
    严格遵循高德MCP协议的15个标准工具名称。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MCPClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        if not hasattr(self, 'initialized'):
            # 优先从参数获取，其次从环境变量获取
            self.api_key = api_key or os.getenv("GAODE_API_KEY") or "7f8fca28ad6fac251afa1318904b4f56"
            if not self.api_key:
                raise ValueError("高德 API Key 未提供。请在环境变量中设置 GAODE_API_KEY 或在初始化时传入。")
            
            self.web_client = GaodeWebAPIClient(api_key=self.api_key, timeout=timeout)
            
            # ✅ 严格按照高德MCP协议注册所有15个官方工具
            self._tool_methods = {
                # 路线规划类 (4个)
                "maps_direction_driving": self.web_client.maps_direction_driving,
                "maps_direction_walking": self.web_client.maps_direction_walking,
                "maps_direction_bicycling": self.web_client.maps_direction_bicycling,
                "maps_direction_transit_integrated": self.web_client.maps_direction_transit_integrated,
                
                # 地理编码类 (2个)
                "maps_geo": self.web_client.geocode_geo,  # 官方名称
                "maps_regeocode": self.web_client.maps_regeocode,
                
                # 搜索类 (3个)
                "maps_text_search": self.web_client.maps_text_search,
                "maps_around_search": self.web_client.maps_around_search,
                "maps_search_detail": self.web_client.maps_search_detail,
                
                # 辅助功能类 (3个)
                "maps_weather": self.web_client.maps_weather,
                "maps_ip_location": self.web_client.maps_ip_location,
                "maps_distance": self.web_client.maps_distance,
                
                # Schema工具 (2个 - 暂未实现Web API，占位)
                "maps_schema_navi": self._not_implemented,
                "maps_schema_take_taxi": self._not_implemented,
                
                # 个人地图 (1个 - 暂未实现Web API，占位)
                "maps_schema_personal_map": self._not_implemented,
            }
            self.initialized = True
            logger.info(f"✅ 高德MCP客户端初始化完成，加载了 {len(self._tool_methods)} 个官方工具")
            logger.info(f"📋 可用工具: {list(self._tool_methods.keys())}")

    async def _not_implemented(self, **kwargs) -> Dict[str, Any]:
        """暂未实现的工具占位符"""
        return {
            "success": False,
            "error": "该工具暂未实现Web API版本，请使用高德地图App的URI Scheme功能"
        }
    
    async def get_tool_methods(self) -> Dict[str, Any]:
        """获取已加载的工具方法字典。"""
        return self._tool_methods

    async def get_tools_metadata(self) -> List[Dict[str, Any]]:
        """
        为 LLM 生成工具的元数据（Schema）。
        严格对应高德MCP协议的15个官方工具。
        """
        metadata = [
            # ========== 路线规划类 ==========
            {
                "name": "maps_direction_driving",
                "description": "高德驾车路线规划。计算两点间的驾车路线，返回距离、耗时和详细步骤。起终点必须是经纬度坐标'lng,lat'格式。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "起点经纬度，格式'lng,lat'，如'116.481028,39.989643'"},
                        "destination": {"type": "string", "description": "终点经纬度，格式'lng,lat'"},
                    },
                    "required": ["origin", "destination"]
                }
            },
            {
                "name": "maps_direction_walking",
                "description": "高德步行路线规划。计算两点间的步行路线。起终点必须是经纬度坐标。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "起点经纬度'lng,lat'"},
                        "destination": {"type": "string", "description": "终点经纬度'lng,lat'"},
                    },
                    "required": ["origin", "destination"]
                }
            },
            {
                "name": "maps_direction_bicycling",
                "description": "高德骑行路线规划。计算两点间的骑行路线。起终点必须是经纬度坐标。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "起点经纬度'lng,lat'"},
                        "destination": {"type": "string", "description": "终点经纬度'lng,lat'"},
                    },
                    "required": ["origin", "destination"]
                }
            },
            {
                "name": "maps_direction_transit_integrated",
                "description": "高德公交路线规划。计算两点间的公交换乘方案。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "起点经纬度'lng,lat'"},
                        "destination": {"type": "string", "description": "终点经纬度'lng,lat'"},
                        "city": {"type": "string", "description": "城市名称或adcode"},
                    },
                    "required": ["origin", "destination", "city"]
                }
            },
            
            # ========== 地理编码类 ==========
            {
                "name": "maps_geo",
                "description": "高德地理编码（官方工具名）。将结构化地址转换为经纬度坐标。返回格式包含lng和lat字段。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "结构化地址，如'北京市海淀区中关村'"},
                        "city": {"type": "string", "description": "城市限定，可选"},
                    },
                    "required": ["address"]
                }
            },
            {
                "name": "maps_regeocode",
                "description": "高德逆地理编码。将经纬度坐标转换为详细地址。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "经纬度坐标'lng,lat'"},
                    },
                    "required": ["location"]
                }
            },
            
            # ========== 搜索类 ==========
            {
                "name": "maps_text_search",
                "description": "高德POI文本搜索。根据关键词搜索地点，返回POI列表（含lng/lat字段）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "string", "description": "搜索关键词，如'咖啡馆'"},
                        "city": {"type": "string", "description": "城市限定，可选"},
                    },
                    "required": ["keywords"]
                }
            },
            {
                "name": "maps_around_search",
                "description": "高德周边搜索。搜索指定坐标周边的POI。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "string", "description": "搜索关键词"},
                        "location": {"type": "string", "description": "中心点经纬度'lng,lat'"},
                        "radius": {"type": "integer", "description": "搜索半径（米），默认1000"},
                    },
                    "required": ["keywords", "location"]
                }
            },
            {
                "name": "maps_search_detail",
                "description": "高德POI详情查询。根据POI ID查询详细信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "POI的唯一标识ID"},
                    },
                    "required": ["id"]
                }
            },
            
            # ========== 辅助功能类 ==========
            {
                "name": "maps_weather",
                "description": "高德天气查询。查询指定城市的天气预报。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称或adcode"},
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "maps_ip_location",
                "description": "高德IP定位。根据IP地址获取地理位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ip": {"type": "string", "description": "IP地址，可选（不填则使用当前请求IP）"},
                    },
                    "required": []
                }
            },
            {
                "name": "maps_distance",
                "description": "高德距离测量。计算多个起点到终点的距离。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origins": {"type": "string", "description": "起点坐标，多个用|分隔，如'lng1,lat1|lng2,lat2'"},
                        "destination": {"type": "string", "description": "终点坐标'lng,lat'"},
                    },
                    "required": ["origins", "destination"]
                }
            },
        ]
        return metadata

    # 移除 initialize 和 get_tools 方法，因为它们的功能已被 __init__ 和 get_tool_methods/get_tools_metadata 替代
