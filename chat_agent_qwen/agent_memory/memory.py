import os
import json
import hashlib
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MemoryManager:
    """
    记忆管理器 - 负责用户对话历史和成功案例的持久化存储
    
    功能特性:
    1. 长期记忆: 按userid分类存储成功案例到独立文件
    2. 短期记忆: 保存当前会话的完整对话历史
    3. 会话管理: 支持临时清空和物理删除
    4. 用户数据隔离: 所有用户数据存储在独立的 user_data/{user_id} 目录下
    """
    def __init__(self, base_dir: str = "./user_data"):
        """初始化记忆存储目录
        
        Args:
            base_dir: 所有用户数据的根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.temp_clear_flags = set()
        self.existing_users = self._load_existing_users()
        logger.info(f"MemoryManager 初始化完成 | 用户数据根目录: {self.base_dir} | 已知用户: {len(self.existing_users)}")

    def _get_user_dir(self, user_id: str) -> Path:
        """获取指定用户的专属数据目录"""
        # 使用 user_id 的哈希值作为目录名，避免特殊字符问题
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()
        user_dir = self.base_dir / hashed_id
        user_dir.mkdir(exist_ok=True)
        
        # 在用户目录中创建子目录
        (user_dir / "files").mkdir(exist_ok=True)
        (user_dir / "success_cases").mkdir(exist_ok=True)
        
        return user_dir

    def _load_existing_users(self) -> set:
        """从目录结构加载所有已知的用户ID哈希"""
        existing = set()
        if not self.base_dir.exists():
            return existing
        for user_dir in self.base_dir.iterdir():
            if user_dir.is_dir():
                existing.add(user_dir.name)
        return existing

    def _get_history_file_path(self, user_id: str) -> Path:
        """获取用户对话历史文件的路径"""
        user_dir = self._get_user_dir(user_id)
        return user_dir / "history.json"

    def is_new_user(self, user_id: str) -> bool:
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()
        return hashed_id not in self.existing_users

    def set_temp_clear(self, user_id: str, enable: bool = True):
        if enable:
            self.temp_clear_flags.add(user_id)
            logger.info(f"为用户 {user_id} 设置临时清空标记。")
        else:
            self.temp_clear_flags.discard(user_id)
            logger.info(f"为用户 {user_id} 取消临时清空标记。")

    def save_memory(self, user_id: str, memory: Dict[str, Any]):
        file_path = self._get_history_file_path(user_id)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2, ensure_ascii=False)
            logger.info(f"用户 {user_id} 的记忆已保存至 {file_path}")
        except Exception as e:
            logger.error(f"保存用户 {user_id} 记忆失败: {e}")
            raise
        
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()
        self.existing_users.add(hashed_id)

    def load_memory(self, user_id: str) -> Dict[str, Any]:
        file_path = self._get_history_file_path(user_id)
        if not file_path.exists():
            return self.default_memory()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                memory = json.load(f)

            if user_id in self.temp_clear_flags:
                memory["conversation_history"] = []
                logger.info(f"用户 {user_id} 触发临时清空标记，历史已清空。")

            return memory
        except Exception as e:
            logger.error(f"加载用户 {user_id} 记忆失败: {e}")
            return self.default_memory()

    def default_memory(self) -> Dict[str, Any]:
        return {"conversation_history": [], "first_visit": True, "files": {}}

    def update_history(self, user_id: str, message: Dict[str, str]):
        memory = self.load_memory(user_id)
        memory["conversation_history"].append(message)
        memory["first_visit"] = False
        self.save_memory(user_id, memory)

    def add_file_reference(self, user_id: str, file_type: str, file_name: str, description: str) -> Path:
        """
        为用户添加一个文件引用，并返回该文件的绝对路径。
        文件将存储在 user_data/{user_id}/files/ 目录下。
        """
        user_dir = self._get_user_dir(user_id)
        files_dir = user_dir / "files"
        
        # 使用安全的文件名
        safe_filename = Path(file_name).name
        file_path = files_dir / safe_filename
        
        memory = self.load_memory(user_id)
        if "files" not in memory:
            memory["files"] = {}
        
        memory["files"][file_name] = {
            "type": file_type,
            "path": str(file_path.relative_to(self.base_dir)), # 存储相对路径
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        self.save_memory(user_id, memory)
        logger.info(f"为用户 {user_id} 添加文件引用: {file_name} -> {file_path}")
        return file_path

    def get_user_files(self, user_id: str) -> Dict[str, Any]:
        """获取用户的所有文件引用"""
        memory = self.load_memory(user_id)
        return memory.get("files", {})

    def save_success_case(self, user_id: str, case_data: Dict[str, Any]):
        """保存成功案例到用户的长期记忆库"""
        user_dir = self._get_user_dir(user_id)
        success_file = user_dir / "success_cases" / "cases.json"
        
        try:
            if success_file.exists():
                with open(success_file, "r", encoding="utf-8") as f:
                    cases = json.load(f)
            else:
                cases = {"user_id": user_id, "cases": []}
            
            from datetime import datetime
            case_data["timestamp"] = case_data.get("timestamp", datetime.now().isoformat())
            cases["cases"].append(case_data)
            cases["cases"] = cases["cases"][-100:] # 保留最近100个
            
            with open(success_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 用户 {user_id} 的成功案例已保存，当前共 {len(cases['cases'])} 个案例")
        except Exception as e:
            logger.error(f"❌ 保存成功案例失败: {e}")
    
    def load_success_cases(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """加载用户的成功案例历史"""
        user_dir = self._get_user_dir(user_id)
        success_file = user_dir / "success_cases" / "cases.json"
        
        if not success_file.exists():
            return []
        
        try:
            with open(success_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("cases", [])[-limit:]
        except Exception as e:
            logger.error(f"❌ 加载成功案例失败: {e}")
            return []

    def clear_context(self, user_id: str):
        """物理删除用户的对话历史文件(保留其他文件和成功案例)"""
        history_file = self._get_history_file_path(user_id)
        
        if history_file.exists():
            try:
                # 只重置历史文件，不删除整个目录
                self.save_memory(user_id, self.default_memory())
                logger.info(f"✅ 用户 {user_id} 的对话历史已清空: {history_file}")
            except Exception as e:
                logger.error(f"❌ 清空用户 {user_id} 的对话历史失败: {e}")
        else:
            logger.info(f"⚠️ 用户 {user_id} 无对话历史可清除。")

        self.temp_clear_flags.discard(user_id)
        logger.info(f"📦 用户 {user_id} 的其他数据已保留。")
