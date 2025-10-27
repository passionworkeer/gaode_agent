# chat_agent_qwen/ICL_agent/icl_agent.py

"""
ICL（In-Context Learning）Agent 模块

该模块实现了上下文学习代理，通过预设的示例对话/任务，引导大模型进行
few-shot 或 zero-shot 推理。它作为主 Agent 的一个可选组件，用于处理
那些不需要复杂任务规划和工具调用的普通对话或特定领域的简单查询。

此模块与主 Agent 的 "核心逻辑层" 协同工作，根据 `use_icl` 参数决定
是否启用 ICL 模式。ICL 示例的质量直接影响模型的推理效果。
"""

from typing import List, Dict, Any, Optional
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 可以根据需要配置日志处理器（如写入文件、控制台输出）
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

class ICLAgent:
    """
    ICL（In-Context Learning）代理类。

    该类负责管理预设的示例对话，并在主 Agent 需要时，利用这些示例
    引导大模型生成更符合预期的回复。
    """

    def __init__(self, model, examples: Optional[List[Dict[str, str]]] = None):
        """
        初始化 ICLAgent。

        Args:
            model: 一个大语言模型实例，必须具有 `generate` 方法。
                   该方法接收一个消息列表（格式为 [{"role": "user/assistant", "content": "..."}]）
                   并返回生成的文本。
            examples (Optional[List[Dict[str, str]]]): 可选的初始示例列表。
        """
        self.model = model
        # 存储上下文示例，格式为 [{"input": "用户输入", "output": "期望输出"}, ...]
        self.examples: List[Dict[str, str]] = examples or []
        logger.info(f"ICLAgent 初始化完成，加载了 {len(self.examples)} 个示例。")

    def add_example(self, example: Dict[str, str]) -> bool:
        """
        添加一个上下文示例。

        Args:
            example (Dict[str, str]): 示例字典，必须包含 "input" 和 "output" 键。

        Returns:
            bool: 如果示例格式正确并成功添加则返回 True，否则返回 False。
        """
        if not isinstance(example, dict) or "input" not in example or "output" not in example:
            logger.warning(f"尝试添加格式错误的 ICL 示例: {example}")
            return False

        self.examples.append(example)
        logger.info(f"成功添加 ICL 示例: {example['input'][:50]}...") # 记录前50个字符
        return True

    def remove_example(self, index: int) -> bool:
        """
        根据索引移除一个上下文示例。

        Args:
            index (int): 要移除的示例的索引。

        Returns:
            bool: 如果成功移除则返回 True，如果索引无效则返回 False。
        """
        if 0 <= index < len(self.examples):
            removed_example = self.examples.pop(index)
            logger.info(f"移除 ICL 示例 (索引 {index}): {removed_example['input'][:50]}...")
            return True
        else:
            logger.warning(f"尝试移除无效索引 {index} 的 ICL 示例，当前列表长度为 {len(self.examples)}")
            return False

    def get_examples(self) -> List[Dict[str, str]]:
        """
        获取当前所有上下文示例。

        Returns:
            List[Dict[str, str]]: 当前的示例列表。
        """
        logger.debug(f"获取 ICL 示例列表，共 {len(self.examples)} 个。")
        return self.examples

    def clear_examples(self):
        """清空所有上下文示例。"""
        count = len(self.examples)
        self.examples = []
        logger.info(f"已清空 {count} 个 ICL 示例。")

    def load_examples_from_config(self, config_path: str) -> bool:
        """
        (预留方法) 从配置文件加载上下文示例。

        Args:
            config_path (str): 配置文件路径（如 JSON 或 YAML 文件）。

        Returns:
            bool: 加载成功返回 True，失败返回 False。
        """
        # TODO: 实现从文件读取逻辑
        # 例如，读取 JSON 文件 [{"input": "...", "output": "..."}, ...]
        logger.warning(f"load_examples_from_config 方法尚未实现，配置路径: {config_path}")
        return False

    def infer(self, user_input: str) -> Optional[str]:
        """
        基于上下文示例和用户输入进行推理。

        此方法构建一个包含示例和用户输入的提示，并将其传递给模型生成回复。
        它是 ICL 功能的核心执行方法，通常在主 Agent 判断需要使用 ICL 时调用。

        Args:
            user_input (str): 用户的输入文本。

        Returns:
            Optional[str]: 模型生成的回复文本。如果过程中发生错误，返回 None。
        """
        if not self.examples:
            logger.warning("ICLAgent 调用 infer 时，没有加载任何示例。")
            # 可以选择返回 None，让主 Agent 流程处理，或者返回一个默认提示
            # return "ICL 示例库为空，无法进行推理。"
            return None

        try:
            # 构建提示词：先添加所有示例，最后添加用户当前输入
            prompt = []
            for example in self.examples:
                prompt.append({"role": "user", "content": example["input"]})
                prompt.append({"role": "assistant", "content": example["output"]})
            prompt.append({"role": "user", "content": user_input})

            logger.debug(f"ICLAgent 构建的提示词: {prompt}")

            # 调用模型生成回复
            # 假设 model.generate 方法接收消息列表并返回字符串
            response = self.model.generate(prompt) # type: ignore

            logger.info(f"ICLAgent 成功生成回复，输入: '{user_input[:30]}...'") # 记录输入前30个字符
            return response

        except AttributeError as e:
            logger.error(f"ICLAgent 调用模型的 generate 方法时出错 (模型可能没有此方法): {e}")
            return None
        except Exception as e:
            logger.error(f"ICLAgent 执行推理时发生未知错误: {e}")
            return None

