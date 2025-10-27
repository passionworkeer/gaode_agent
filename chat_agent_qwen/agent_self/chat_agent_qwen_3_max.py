from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import SecretStr
from typing import AsyncIterator, Optional
import asyncio
import os
from dotenv import load_dotenv

# 在模块级别加载环境变量，确保任何时候导入此模块时都已加载
load_dotenv()

# 定义自定义流式回调类
class PrintAndStoreHandler(BaseCallbackHandler):
    """在控制台实时打印token，并保存完整回复"""
    def __init__(self):
        self.current_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.current_text += token

class QwenModel:
    def __init__(self):
        # 初始化一个通用的流式模型
        self.handler = PrintAndStoreHandler()
        # 从环境变量获取API密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")

        self.llm = ChatOpenAI(
            model="qwen3-max",  # 默认使用 qwen3-max
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=SecretStr(api_key),  # 使用从环境变量获取的密钥
            streaming=True,  # 允许流式
            callbacks=[self.handler],  # 回调统一在这里注册
            temperature=0.7,
        )

    def generate(self, messages, **kwargs):
        """普通调用(返回完整文本,不实时打印)"""
        response = self.llm.invoke(messages, **kwargs)
        return response.content

    def stream_generate(self, messages):
        """流式输出(实时打印并返回完整回复)"""
        # 每次调用前重置handler缓存
        self.handler.current_text = ""
        self.llm.invoke(messages)
        return self.handler.current_text
    
    async def astream_generate(self, messages) -> AsyncIterator[str]:
        """异步流式生成 - 逐token返回
        
        Args:
            messages: 消息列表
        
        Yields:
            单个token字符串
        """
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    async def agenerate(self, messages, **kwargs) -> str:
        """异步生成 - 返回完整文本
        
        Args:
            messages: 消息列表
        
        Returns:
            完整回复文本
        """
        full_text = ""
        async for token in self.astream_generate(messages):
            full_text += token
        return full_text

    def get_model(self):
        """获取底层LangChain模型对象"""
        return self.llm
