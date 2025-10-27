# chat_agent_qwen/utils/message_validator.py
"""
LLM消息格式验证器 - 防止MESSAGE_COERCION_FAILURE错误

功能:
1. 验证消息列表的合法性
2. 自动过滤 Ellipsis (...) 和 None
3. 修复常见格式错误
4. 提供详细的错误日志
"""

import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


class MessageValidator:
    """LLM消息格式验证器 - 确保所有消息符合LangChain格式要求"""
    
    VALID_ROLES = {"system", "user", "assistant", "function"}
    
    @classmethod
    def validate_messages(cls, messages: Any) -> List[Dict[str, str]]:
        """验证并修复消息列表
        
        Args:
            messages: 原始消息列表
            
        Returns:
            验证后的消息列表
            
        Raises:
            ValueError: 消息格式无法修复时抛出
        """
        if not isinstance(messages, list):
            raise ValueError(f"messages必须是list类型，当前类型: {type(messages)}")
        
        validated = []
        for i, msg in enumerate(messages):
            # 过滤掉 Ellipsis (...)
            if msg is Ellipsis or msg == ...:
                logger.warning(f"⚠️ 消息索引 {i} 为 Ellipsis，已跳过")
                continue
            
            # 过滤掉 None
            if msg is None:
                logger.warning(f"⚠️ 消息索引 {i} 为 None，已跳过")
                continue
            
            # 验证消息结构
            if not isinstance(msg, dict):
                logger.error(f"❌ 消息索引 {i} 不是dict类型: {type(msg)}")
                raise ValueError(f"消息必须是dict，当前: {type(msg)}")
            
            # 验证必需字段
            if "role" not in msg:
                logger.error(f"❌ 消息索引 {i} 缺少role字段: {msg}")
                raise ValueError(f"消息缺少role字段")
            
            if "content" not in msg:
                logger.error(f"❌ 消息索引 {i} 缺少content字段: {msg}")
                raise ValueError(f"消息缺少content字段")
            
            # 验证 role 值
            role = msg["role"]
            if role not in cls.VALID_ROLES:
                logger.warning(f"⚠️ 消息索引 {i} role无效: {role}，已修正为 'user'")
                role = "user"
            
            # 验证 content 类型
            content = msg["content"]
            if content is None:
                logger.warning(f"⚠️ 消息索引 {i} content为None，已替换为空字符串")
                content = ""
            elif not isinstance(content, str):
                logger.warning(f"⚠️ 消息索引 {i} content不是str: {type(content)}，已转换")
                content = str(content)
            
            # 添加验证后的消息
            validated.append({"role": role, "content": content})
        
        if not validated:
            raise ValueError("验证后的消息列表为空")
        
        logger.info(f"✅ 消息列表验证通过，共 {len(validated)} 条消息")
        return validated
    
    @classmethod
    def safe_extend_history(cls, prompt: List[Dict], history: List[Dict], max_count: int = 5) -> List[Dict]:
        """安全地将历史记录添加到prompt中
        
        Args:
            prompt: 当前prompt列表
            history: 历史消息列表
            max_count: 最多添加的历史消息数量
            
        Returns:
            合并后的消息列表
        """
        # 验证 prompt
        validated_prompt = cls.validate_messages(prompt)
        
        # 验证并截取历史记录
        if history:
            try:
                validated_history = cls.validate_messages(history)
                selected_history = validated_history[-max_count:] if len(validated_history) > max_count else validated_history
                validated_prompt.extend(selected_history)
            except ValueError as e:
                logger.warning(f"历史记录验证失败，已忽略: {e}")
        
        return validated_prompt
    
    @classmethod
    def safe_append(cls, prompt: List[Dict], role: str, content: str) -> List[Dict]:
        """安全地向prompt添加一条消息
        
        Args:
            prompt: 当前prompt列表
            role: 消息角色
            content: 消息内容
            
        Returns:
            添加后的消息列表
        """
        if role not in cls.VALID_ROLES:
            logger.warning(f"role '{role}' 无效，已修正为 'user'")
            role = "user"
        
        if content is None:
            logger.warning("content为None，已替换为空字符串")
            content = ""
        elif not isinstance(content, str):
            logger.warning(f"content不是str: {type(content)}，已转换")
            content = str(content)
        
        prompt.append({"role": role, "content": content})
        return prompt
