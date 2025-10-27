# chat_agent_qwen/utils/security.py
"""
安全工具模块 - 提供文件路径验证和代码安全检查

功能:
1. 文件路径白名单验证 (防止路径遍历攻击)
2. 文件名清理 (移除危险字符)
3. LLM生成代码安全检查
"""

from pathlib import Path
from typing import Literal, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """安全相关错误"""
    pass


class SecureFileManager:
    """安全的文件管理器"""

    # 白名单目录 (使用绝对路径)
    BASE_DIR = Path(__file__).parent.parent.parent  # 项目根目录
    ALLOWED_DIRS = {
        "temp_visualizations": (BASE_DIR / "temp_visualizations").resolve(),
        "temp_files": (BASE_DIR / "temp_files").resolve(),
        "memory": (BASE_DIR / "memory").resolve()
    }

    @classmethod
    def ensure_dirs_exist(cls):
        """确保所有白名单目录存在"""
        for dir_name, dir_path in cls.ALLOWED_DIRS.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"目录已确保存在: {dir_path}")

    @classmethod
    def validate_path(
        cls,
        file_path: str,
        category: Literal["temp_visualizations", "temp_files", "memory"]
    ) -> Path:
        """验证文件路径在允许范围内

        Args:
            file_path: 待验证的路径 (相对或绝对)
            category: 文件类别,必须是白名单中的key

        Returns:
            规范化的Path对象 (绝对路径)

        Raises:
            SecurityError: 路径不在白名单内或包含危险字符
        """
        # 确保目录存在
        cls.ensure_dirs_exist()

        # 获取白名单基础目录
        if category not in cls.ALLOWED_DIRS:
            raise SecurityError(f"未知的文件类别: {category}")

        allowed_base = cls.ALLOWED_DIRS[category]

        # 将输入路径转换为绝对路径
        if Path(file_path).is_absolute():
            target = Path(file_path).resolve()
        else:
            # 相对路径: 基于白名单目录解析
            target = (allowed_base / file_path).resolve()

        # 检查是否在白名单目录内
        try:
            target.relative_to(allowed_base)
        except ValueError:
            raise SecurityError(
                f"路径 '{file_path}' 不在允许的 '{category}' 目录内\n"
                f"目标路径: {target}\n"
                f"允许基础路径: {allowed_base}"
            )

        # 检查路径遍历攻击 (双重保险)
        if ".." in str(file_path):
            raise SecurityError(f"路径 '{file_path}' 包含非法字符 '..'")

        logger.info(f"路径验证通过: {target} (类别: {category})")
        return target

    @classmethod
    def sanitize_filename(cls, filename: str, max_length: int = 200) -> str:
        """清理文件名中的危险字符

        Args:
            filename: 原始文件名
            max_length: 最大长度限制

        Returns:
            清理后的安全文件名
        """
        # 1. 仅保留字母、数字、下划线、连字符、点
        safe_name = re.sub(r'[^\w\-.]', '_', filename)

        # 2. 移除多个连续点 (防止 ../)
        safe_name = re.sub(r'\.{2,}', '.', safe_name)

        # 3. 移除开头的点 (隐藏文件)
        safe_name = safe_name.lstrip('.')

        # 4. 限制长度
        if len(safe_name) > max_length:
            # 保留扩展名
            name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            name = name[:max_length - len(ext) - 1]
            safe_name = f"{name}.{ext}" if ext else name

        # 5. 确保非空
        if not safe_name:
            safe_name = "unnamed_file"

        logger.debug(f"文件名清理: '{filename}' → '{safe_name}'")
        return safe_name

    @classmethod
    def get_safe_path(cls, filename: str, category: Literal["temp_visualizations", "temp_files", "memory"]) -> Path:
        """生成安全的文件路径

        Args:
            filename: 文件名
            category: 文件类别

        Returns:
            安全的绝对路径
        """
        safe_filename = cls.sanitize_filename(filename)
        return cls.validate_path(safe_filename, category)


class CodeSecurityChecker:
    """LLM生成代码安全检查器"""

    # 危险模式定义 (模式, 描述)
    DANGEROUS_PATTERNS = [
        (r'__import__\s*\(', "禁止动态导入 (__import__)"),
        (r'open\s*\([^)]*["\'][wax]', "禁止写入文件 (open with 'w'/'a'/'x')"),
        (r'os\.(system|popen|exec|spawn|fork)', "禁止执行系统命令"),
        (r'subprocess\.(?!PIPE|STDOUT)', "禁止调用子进程 (subprocess)"),
        (r'eval\s*\(', "禁止 eval()"),
        (r'exec\s*\(', "禁止 exec()"),
        (r'compile\s*\(', "禁止 compile()"),
        (r'import\s+(socket|urllib|requests|httpx)', "禁止网络请求库"),
        (r'import\s+(shutil|pathlib\.Path\(\)\.unlink)', "禁止文件操作库"),
        (r'\.unlink\(', "禁止删除文件 (.unlink())"),
        (r'\.rmdir\(', "禁止删除目录 (.rmdir())"),
        (r'globals\(\)', "禁止访问全局变量 (globals())"),
        (r'locals\(\)', "禁止访问局部变量 (locals())"),
        (r'vars\(\)', "禁止访问变量字典 (vars())"),
        (r'dir\(\)', "禁止内省 (dir())"),
        (r'__[a-z]+__', "禁止访问魔术方法/属性"),
    ]

    @classmethod
    def check_code_safety(cls, code: str) -> Tuple[bool, str]:
        """检查代码安全性

        Args:
            code: 待检查的Python代码

        Returns:
            (is_safe, reason): 是否安全及原因描述
        """
        for pattern, reason in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                logger.warning(f"代码安全检查失败: {reason}\n匹配模式: {pattern}")
                return False, reason

        logger.info("代码安全检查通过")
        return True, "代码安全"

    @classmethod
    def get_allowed_imports(cls) -> list:
        """获取允许导入的模块列表"""
        return [
            "math", "datetime", "json", "random", "re", "collections",
            "itertools", "functools", "operator", "typing",
            "matplotlib", "matplotlib.pyplot",
            "reportlab", "reportlab.lib", "reportlab.platypus", "reportlab.pdfgen",
            "openpyxl", "openpyxl.styles", "openpyxl.utils"
        ]

    @classmethod
    def validate_imports(cls, code: str) -> Tuple[bool, str]:
        """验证代码中的导入语句是否在白名单内

        Args:
            code: Python代码

        Returns:
            (is_valid, reason): 是否有效及原因
        """
        allowed = cls.get_allowed_imports()

        # 提取所有 import 语句
        import_pattern = r'^\s*(?:from\s+([\w\.]+)\s+)?import\s+([\w\.\*\s,]+)'
        imports = re.findall(import_pattern, code, re.MULTILINE)

        for from_module, import_names in imports:
            # 处理 from X import Y 和 import X
            module = from_module if from_module else import_names.split(',')[0].strip()

            # 检查是否在白名单内 (支持前缀匹配,如 matplotlib.pyplot 匹配 matplotlib)
            if not any(module.startswith(allowed_mod) for allowed_mod in allowed):
                reason = f"禁止导入模块: {module} (未在白名单中)"
                logger.warning(reason)
                return False, reason

        logger.info("代码导入语句验证通过")
        return True, "导入语句合法"


# 初始化时确保目录存在
SecureFileManager.ensure_dirs_exist()
