# chat_agent_qwen/utils/step_context.py
"""
步骤执行上下文管理器 - 实现多步骤工具调用的结果传递机制

功能:
1. 存储每个步骤的执行结果
2. 替换参数中的占位符 (如 {step_0_result.location})
3. 支持嵌套字段访问和列表索引
"""

import re
import logging
from typing import Any, Dict, Optional, List
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """任务执行失败时的策略"""
    FAIL_FAST = "fail_fast"  # 快速失败
    GRACEFUL_DEGRADE = "graceful_degrade"  # 优雅降级

class TaskStep(BaseModel):
    """单个任务步骤的数据模型"""
    goal: str = Field(..., description="此步骤要实现的具体目标")
    tool_name: str = Field(..., description="要调用的工具名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="传递给工具的参数")


class StepContext:
    """执行上下文 - 管理步骤间数据流"""
    def __init__(self):
        self.results: Dict[int, Any] = {}  # {step_index: result_data}
        self.metadata: Dict[str, Any] = {}  # 全局元数据
        self.step_count = 0

    def set_result(self, step_index: int, result: Any):
        """保存步骤结果并自动提取元数据
        Args:
            step_index: 步骤索引
            result: 步骤执行结果 (可以是dict, str, list等)
        """
        self.results[step_index] = result
        self.step_count = max(self.step_count, step_index + 1)

        # 自动提取常用字段到元数据 (方便快速访问)
        if isinstance(result, dict):
            # 高德地图相关字段
            if "location" in result:
                self.metadata[f"step_{step_index}_location"] = result["location"]
            if "formatted_address" in result:
                self.metadata[f"step_{step_index}_address"] = result["formatted_address"]
            if "distance" in result:
                self.metadata[f"step_{step_index}_distance"] = result["distance"]
            if "duration" in result:
                self.metadata[f"step_{step_index}_duration"] = result["duration"]

        logger.debug(f"步骤 {step_index} 结果已保存: {str(result)[:100]}...")

    def get_result(self, step_index: int) -> Optional[Any]:
        """获取指定步骤的结果"""
        return self.results.get(step_index)

    def replace_placeholders(self, params: Dict[str, Any], current_step: int) -> Dict[str, Any]:
        """递归替换参数中的占位符
        
        支持格式:
        - {step_0_result} → 完整结果
        - {step_0_result.location} → 结果中的location字段
        - {step_0_result[0].name} → 列表索引 + 字段访问
        - {step_0_result.data.province} → 嵌套字段访问
        
        Args:
            params: 原始参数字典
            current_step: 当前步骤索引 (用于防止引用未来步骤)
        
        Returns:
            替换后的参数字典
        
        Raises:
            ValueError: 引用了未来步骤或不存在的步骤
        """
        def resolve_path(obj: Any, path: str) -> Any:
            """解析路径: location.province 或 [0].name
            
            Args:
                obj: 要解析的对象
                path: 字段路径,如 "location" 或 "[0].name" 或 "data.province"
            
            Returns:
                解析后的值
            
            ⚠️ 重要修复:
                - 增加列表索引越界检查
                - 增加字典键存在性检查
                - **严格禁止**字符串方法调用（如 .split()）
            """
            if not path:
                return obj

            # ✅ 严格检测并拒绝任何方法调用或算术/比较表达式
            if re.search(r'\.[A-Za-z_]\w*\s*\(', path):
                # 特化提示: 常见的 .split 用法
                if ".split(" in path:
                    msg = (
                        f"不支持的方法调用: '{path}'。禁止使用 .split() 解析坐标; 请直接引用已注入的独立字段 lng/lat, 例如 "
                        f"{{step_X_result.geocodes[0].lng}} / {{step_X_result.geocodes[0].lat}} 或 {{step_X_result.pois[0].lng}} / {{step_X_result.pois[0].lat}}。"
                    )
                else:
                    msg = (
                        f"不支持的方法调用或复杂表达式: '{path}'。占位符仅允许字段/下标访问, 禁止任何方法/函数(如 .get(), int())。"
                    )
                logger.error("❌ " + msg)
                raise ValueError(msg)

            # 禁止算术/比较运算，包括 / * + - % > < 等
            if re.search(r'[\*/%]|(?<!e)[\+\-]|[<>]', path):
                msg = (
                    f"不支持的路径表达式: '{path}'。禁止在占位符中进行任何运算; 如需公里/分钟等, 请使用已提供的字段(如 distance_km, duration_min)。"
                )
                logger.error("❌ " + msg)
                raise ValueError(msg)
            
            parts = re.split(r'\.|\[|\]', path)
            parts = [p for p in parts if p]  # 移除空字符串
            # 容错包装: 如果外层对象是单条结果（包含 location/distance/steps 等），
            # 但占位符使用了 results/paths/pois 形式（如 results[0].location），
            # 我们尝试把 obj 包装成 {results: [obj]} 或 {paths: [obj]} 以兼容旧格式。
            if parts:
                first_key = parts[0]
                try:
                    if isinstance(obj, dict) and first_key in ('results', 'paths', 'pois') and first_key not in obj:
                        # 若顶层已有类似字段，则进行映射
                        if first_key == 'results':
                            if 'pois' in obj and isinstance(obj['pois'], list):
                                logger.debug("🔁 step_context: 将顶层 'pois' 映射为 'results' 以兼容占位符解析")
                                obj = {'results': obj['pois']}
                            elif any(k in obj for k in ('location', 'formatted_address', 'province', 'city', 'district')):
                                logger.debug("🔁 step_context: 将单条结果包装为 'results' 列表以兼容占位符解析")
                                obj = {'results': [obj]}
                        elif first_key == 'paths':
                            if 'paths' not in obj and 'routes' in obj and isinstance(obj['routes'], list):
                                logger.debug("🔁 step_context: 将顶层 'routes' 映射为 'paths' 以兼容占位符解析")
                                obj = {'paths': obj['routes']}
                            elif any(k in obj for k in ('steps', 'distance', 'duration')):
                                logger.debug("🔁 step_context: 将单条路线结果包装为 'paths' 列表以兼容占位符解析")
                                obj = {'paths': [obj]}
                        elif first_key == 'pois':
                            if 'pois' not in obj and 'results' in obj and isinstance(obj['results'], list):
                                logger.debug("🔁 step_context: 将顶层 'results' 映射为 'pois' 以兼容占位符解析")
                                obj = {'pois': obj['results']}
                except Exception as e:
                    logger.debug(f"step_context: 容错包装时出错: {e}")

            current = obj
            for i, part in enumerate(parts):
                try:
                    if part.isdigit():
                        # ✅ 修复: 数组索引前检查长度
                        index = int(part)
                        if isinstance(current, list):
                            if index < len(current):
                                current = current[index]
                            else:
                                logger.warning(f"⚠️ 列表索引越界: {path} (索引 {index}, 长度 {len(current)})")
                                return None
                        else:
                            logger.warning(f"⚠️ 尝试对非列表对象使用索引: {path} (当前类型: {type(current)})")
                            return None
                    else:
                        # ✅ 修复: 字段访问时安全检查
                        if isinstance(current, dict):
                            if part in current:
                                current = current[part]
                            else:
                                avail = list(current.keys())
                                logger.warning(
                                    f"⚠️ 字典键不存在: 尝试路径='{path}'，缺失键='{part}'，当前节点可用键前5={avail[:5]} (完整键数量={len(avail)})"
                                )
                                return None
                        elif hasattr(current, part):
                            current = getattr(current, part)
                        else:
                            logger.warning(f"⚠️ 对象无此属性: {path} (属性 '{part}', 对象类型 {type(current)})")
                            return None

                    # ✅ 提前检查 None 值
                    if current is None:
                        logger.warning(f"⚠️ 路径 '{path}' 在第 {i+1} 步解析为 None")
                        return None
                        
                except (KeyError, IndexError, TypeError, AttributeError) as e:
                    logger.warning(f"⚠️ 解析路径 '{path}' 在第 {i+1} 步失败: {e}")
                    return None

            return current

        def replace_value(value: Any) -> Any:
            """递归替换单个值"""
            if isinstance(value, str):
                # 查找占位符: {step_N_result.path}
                pattern = r'\{step_(\d+)_result(?:\.([^\}]+))?\}'

                def replacer(match):
                    step_idx = int(match.group(1))
                    path = match.group(2)  # 可能为 None

                    # ✅ 增强修复: 防止引用自身或未来步骤
                    if step_idx >= current_step:
                        error_msg = f"❌ 步骤 {current_step} 不能引用自身或未来步骤 {step_idx} 的结果"
                        logger.error(error_msg)
                        logger.error(f"💡 提示: 工具只能引用前序步骤（step_0 到 step_{current_step-1}）")
                        raise ValueError(error_msg)

                    # 获取步骤结果
                    result = self.results.get(step_idx)
                    if result is None:
                        logger.warning(f"⚠️ 步骤 {step_idx} 结果不存在,返回空字符串 | 当前已保存步骤: {list(self.results.keys())}")
                        return ""

                    # 解析路径
                    if path:
                        resolved = resolve_path(result, path)
                        if resolved is not None:
                            logger.debug(f"✅ 解析成功: step_{step_idx}_result.{path} = {resolved}")
                            return str(resolved)
                        else:
                            logger.warning(f"⚠️ 路径 'step_{step_idx}_result.{path}' 解析为None")
                            return ""
                    else:
                        # 返回完整结果
                        if isinstance(result, (dict, list)):
                            import json
                            return json.dumps(result, ensure_ascii=False)
                        return str(result)

                # 替换所有占位符
                try:
                    return re.sub(pattern, replacer, value)
                except ValueError as e:
                    # 重新抛出,让外部处理
                    raise e

            elif isinstance(value, dict):
                # 递归处理字典
                return {k: replace_value(v) for k, v in value.items()}

            elif isinstance(value, list):
                # 递归处理列表
                return [replace_value(v) for v in value]

            else:
                # 其他类型直接返回
                return value

        try:
            replaced_params = replace_value(params)
            logger.info(f"步骤 {current_step} 参数替换完成: {replaced_params}")
            return replaced_params
        except ValueError as e:
            logger.error(f"参数替换失败: {e}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """获取上下文摘要信息"""
        return {
            "total_steps": self.step_count,
            "completed_steps": len(self.results),
            "metadata": self.metadata
        }

    def clear(self):
        """清空上下文"""
        self.results.clear()
        self.metadata.clear()
        self.step_count = 0
        logger.info("执行上下文已清空")
