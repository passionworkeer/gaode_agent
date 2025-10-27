# chat_agent_qwen/utils/json_parser.py
"""
鲁棒的JSON解析工具 - 处理LLM输出的各种格式问题

功能:
1. 移除Markdown代码块
2. 清理注释 (// 和 /* */)
3. 提取JSON对象
4. 自动修复常见格式错误
5. 清理非法控制字符
"""

import re
import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class RobustJSONParser:
    """鲁棒的JSON解析器,专门处理LLM输出"""

    @staticmethod
    def clean_markdown(text: str) -> str:
        """移除Markdown代码块标记

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 移除 ```json 和 ```
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        return text.strip()

    @staticmethod
    def remove_comments(text: str) -> str:
        """移除JSON中的注释 (// 和 /* */)

        Args:
            text: JSON文本

        Returns:
            移除注释后的文本
        """
        # 移除单行注释 //
        text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)

        # 移除多行注释 /* */
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        return text

    @staticmethod
    def extract_json_object(text: str) -> Optional[str]:
        """从文本中提取第一个JSON对象或数组

        Args:
            text: 可能包含JSON的文本

        Returns:
            提取的JSON字符串,如果未找到则返回None
        """
        # 尝试匹配JSON数组 [...]
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            return array_match.group(0)

        # 尝试匹配JSON对象 {...}
        object_match = re.search(r'\{.*\}', text, re.DOTALL)
        if object_match:
            return object_match.group(0)

        return None

    @staticmethod
    def fix_common_errors(text: str) -> str:
        """修复常见的JSON格式错误

        Args:
            text: JSON文本

        Returns:
            修复后的文本
        """
        # 1. 移除末尾的逗号 (trailing comma)
        text = re.sub(r',\s*([}\]])', r'\1', text)

        # 2. 修复单引号为双引号 (注意: 这可能会误伤字符串内容,需谨慎)
        # text = text.replace("'", '"')

        # 3. 确保键名有引号 (简化处理,可能不完美)
        # 匹配 key: value 并转换为 "key": value
        # text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)

        return text

    @staticmethod
    def remove_control_characters(text: str) -> str:
        """移除JSON字符串中的非法控制字符

        Args:
            text: JSON文本

        Returns:
            移除控制字符后的文本
        """
        # 使用正则表达式移除 ASCII 控制字符 (0x00 - 0x1F 以及 0x7F DEL)
        # \x00-\x1F 匹配 0 到 31
        # \x7F 匹配 127 (DEL)
        cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        return cleaned_text

    @classmethod
    def parse(cls, text: str) -> Optional[Any]:
        """鲁棒解析JSON文本

        Args:
            text: 可能包含JSON的文本 (可能有Markdown、注释等)

        Returns:
            解析后的Python对象 (dict或list),失败返回None
        """
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return None

        # 步骤1: 移除Markdown
        text = cls.clean_markdown(text)

        # 步骤2: 移除注释
        text = cls.remove_comments(text)

        # 步骤3: 提取JSON对象
        json_text = cls.extract_json_object(text)
        if not json_text:
            logger.warning(f"未能提取JSON对象,原始文本: {text[:200]}...")
            return None

        # --- ✅ 新增步骤: 移除控制字符 ---
        json_text = cls.remove_control_characters(json_text)
        # ----------------------------

        # 步骤4: 修复常见错误
        json_text = cls.fix_common_errors(json_text)

        # 步骤5: 尝试解析
        try:
            result = json.loads(json_text)
            logger.info(f"JSON解析成功,类型: {type(result).__name__}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}\n文本片段: {json_text[:500]}...")
            
            # 步骤6: 尝试使用 json_repair (如果安装了)
            try:
                import json_repair
                # json_repair 也可能受控制字符影响，所以最好在它之前就清理
                # 如果 json_repair 内部能处理，这一步是安全的
                result = json_repair.loads(json_text)
                logger.info(f"JSON修复解析成功,类型: {type(result).__name__}")
                return result
            except ImportError:
                logger.warning("json_repair 未安装,无法自动修复JSON")
            except Exception as repair_error:
                logger.error(f"JSON修复也失败: {repair_error}")
            
            return None

    @classmethod
    def parse_with_fallback(cls, text: str, fallback: Any = None) -> Any:
        """带默认值的JSON解析

        Args:
            text: JSON文本
            fallback: 解析失败时的默认值

        Returns:
            解析结果或默认值
        """
        result = cls.parse(text)
        return result if result is not None else fallback