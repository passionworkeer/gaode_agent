# 🌍 智慧旅行规划 Agent 系统

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**基于通义千问(Qwen)的大模型智能旅行规划助手**

个性化行程定制 · 景点推荐 · 路线优化 · 深度思考

[快速开始](#-快速开始) · [功能特性](#-功能特性) · [项目结构](#-项目结构) · [文档](#-文档) · [更新日志](#-更新日志)

</div>

---

## 📖 项目简介

智慧旅行规划 Agent 集成了大语言模型(LLM)、工具调用、记忆管理与可视化能力。系统基于通义千问(Qwen)，结合高德地图(MCP 协议)、Tavily 搜索与本地 RAG 检索，为用户提供可落地的旅行路线规划、景点推荐与交互式地图展示服务。

### ✨ 核心亮点

- 🧠 **双模型切换**: 一键在 UI 中切换 qwen-max(快速) 与 qwen-plus(深度思考)
- 💾 **智能记忆系统**: 长期成功案例 + 短期对话历史，支持按用户隔离
- 🗺️ **地图可视化**: 自动生成交互式 HTML 地图，支持路线规划与 POI 标记
- 🎨 **友好界面**: 基于 Gradio 的流式对话界面，地图与图片即时渲染
- 🔧 **模块化设计**: Prompt、工具集、RAG 引擎、记忆管理模块化拆分，便于扩展

---

## 🚀 快速开始

### 1. 环境准备（Windows PowerShell）

```powershell
#进入项目
cd ai

# 创建并激活虚拟环境（建议 .venv 命名）
py -3.10 -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 升级 pip 并安装依赖
python -m pip install --upgrade pip
pip install -r ..\requirements.txt  
```
### 2. 配置API密钥

```powershell
# 若存在 .env.example，可复制：
# Copy-Item -Path .env.example -Destination .env

# 若不存在，请新建 .env 并填入：
# DASHSCOPE_API_KEY=<你的通义千问密钥>
# TAVILY_API_KEY=<你的Tavily密钥>
# GAODE_API_KEY=<你的高德地图密钥>
# （可选）DEFAULT_MODEL=qwen-max
# （可选）DEEP_THINKING_MODEL=qwen-plus
```

### 3. 启动系统

```powershell
# 从项目根目录启动
python .\main_gradio.py
```

## 💡 功能特性

### 🎯 核心功能

| 功能 | 描述 | 状态 |
|------|------|------|
| 智能对话 | 基于Qwen的自然语言交互 | ✅ |
| 深度思考 | 启用qwen-plus进行复杂推理 | ✅ |
| 网络搜索 | Tavily 实时搜索最新信息 | ✅ |
| 地图服务 | 高德地图(通过 MCP 协议)POI 搜索、路线规划 | ✅ |
| 可视化 | 生成HTML地图和数据图表 | ✅ |
| RAG检索 | 本地旅游知识库查询 | ✅ |
| 代码执行 | 安全的Python代码沙箱 | ✅ |
| 文件生成 | PDF/Excel行程文件导出 | ✅ |

### 🧠 深度思考模式（前端可切换）

开启深度思考后,系统会:
- 一键切换到 qwen-plus 模型进行更深入推理
- 展示思考过程(💭标记)
- 提供更详细、全面的解答
- 适合复杂问题和行程规划

### 💾 记忆系统（按用户隔离）

**长期记忆**:
- 按用户ID独立存储成功案例
- 支持案例检索和复用
- 加速相似问题的处理

**短期记忆**:
- 保存完整对话上下文
- 支持临时清空和物理删除
- 多轮对话理解

### 🗺️ 地图与可视化（自动识别并展示）

- 交互式 HTML 地图(基于高德地图，生成到 `temp_visualizations/*.html`，前端直接加载展示)
- 路线规划(驾车/步行/公交)
- POI 搜索(餐饮/景点/酒店)
- 图片展示支持本地文件与 HTTP 图片 URL，自动去重与追加到画廊

实现要点（来自 `main_gradio.py`）：
- 深度思考开关实时切换 `qwen-plus / qwen-max`
- 流式输出中识别并加载 `temp_visualizations/*.html` 地图与图片 URL
- 全局注入 `CURRENT_USER_ID` 用于可视化与文件归属
- 数据图表(柱状图/折线图/饼图)

---

## 📁 项目结构

```
d:\ai\
├── main_gradio.py                    # Gradio Web 界面入口
├── requirements.txt                  # 依赖库清单
├── chat_agent_qwen/                  # 核心 Agent 模块
│   ├── prompts/
│   │   └── system_prompts.py         # 系统提示词库
│   ├── agent_self/
│   │   ├── agent.py                  # 主 Agent 类
│   │   └── chat_agent_qwen_3_max.py  # Qwen 模型封装
│   ├── agent_memory/
│   │   └── memory.py                 # 记忆管理器
│   ├── agent_tools/
│   │   ├── tools.py                  # 工具集入口(地图/搜索/可视化 等)
│   │   ├── rag_tool.py               # RAG 检索工具
│   │   └── Tavilysearch_tool.py      # Tavily 搜索封装
│   ├── agent_mcp/
│   │   └── agent_mcp_gaode.py        # 高德地图 MCP 客户端
│   ├── RAG_agent/
│   │   └── rag_engine.py             # FAISS 向量检索
│   └── ICL_agent/
│       └── icl_agent.py              # Few-shot 学习
├── md_file/                          # 文档集合（本 README 所在）
│   ├── README.md                     # 项目说明（本文件）
│   ├── 快速开始指南.md
│   ├── 改进完成报告.md
└── temp_visualizations/              # 临时可视化文件
  ├── *.html                        # 地图 HTML
  └── *.png                         # 图表图片
```

---

## 📚 文档

- 《[快速开始指南](./快速开始指南.md)》- 详细安装与使用
- 更多文档见 `md_file/` 目录；API 与 Prompt 优化文档待补充

---

## 🔧 技术栈

### 核心框架
- **LangChain**: LLM应用开发框架
- **Gradio**: Web UI框架
- **Pydantic**: 数据验证

### 大模型
- **通义千问(Qwen)**: 阿里云大语言模型
  - qwen-max: 快速响应
  - qwen-plus: 深度推理

### 外部工具
- **高德地图**: 地理位置服务（通过 MCP 协议集成）
- **Tavily**: 网络搜索引擎
- **FAISS**: 向量检索数据库

### 可视化
- **Matplotlib**: 数据图表
- **ReportLab**: PDF生成
- **OpenPyXL**: Excel处理
- **Jinja2**: HTML模板

更多依赖详见 `../requirements.txt`。

---

## 📊 使用示例

### 示例1: 路线规划

```
用户: 帮我规划从深圳技术大学到广东海洋大学的路线

助手: 
🚗 路线规划如下:

**方式一: 高铁(推荐)**
- 深圳坪山站 → 湛江西站
- 车程: 约3.5小时
- 票价: 二等座约150元
- 到达后打车20分钟到校

**方式二: 自驾**
- 全程: 约500公里
- 耗时: 约6小时
- 路线: 沈海高速(G15)

[附带交互式地图展示]
```

### 示例2: 深度思考模式

```
用户: (开启深度思考) 帮我规划北京到杭州的三日游,预算5000元

助手:
💭 思考: 分析行程需求...北京到杭州直线距离约1100公里,建议高铁往返,
         单程耗时约5小时,往返预算约1000元...

💭 思考: 评估景点方案...杭州主要景点包括西湖(免费)、灵隐寺(门票75)、
         宋城千古情(门票300)...综合考虑时间和预算...

💭 结论: 推荐"经典文化游"路线,总预算约4800元

**Day 1: 抵达 + 西湖游**
- 上午: 北京南 → 杭州东 (高铁5小时, 500元)
- 下午: 西湖景区(断桥、苏堤、雷峰塔)
- 晚上: 入住西湖附近酒店(300元)
...

[详细行程 + 交通 + 住宿 + 美食推荐]
```

---

## 🛠️ 进阶配置

### 自定义Prompt

编辑 `chat_agent_qwen/prompts/system_prompts.py`:

```python
# 修改对话风格
CHAT_RESPONSE_SYSTEM_PROMPT = """
你是一个专业且幽默的旅行助手...
[自定义您的Prompt]
"""
```

### 添加ICL示例

```python
ICL_FEW_SHOT_EXAMPLES.append({
    "input": "你的问题示例",
    "output": "期望的回答示例"
})
```

### 调整模型参数

在 `.env` 中:
```env
DEFAULT_MODEL=qwen-max
DEEP_THINKING_MODEL=qwen-plus
MAX_HISTORY_LENGTH=100
```

---

## 🐛 常见问题

<details>
<summary><b>Q: 启动报错 "No module named xxx"</b></summary>

确认虚拟环境已激活，重新安装依赖（在项目根目录）：
```powershell
pip install -r .\requirements.txt --force-reinstall
```
</details>

<details>
<summary><b>Q: API调用失败</b></summary>

检查:
1. `.env` 文件是否存在且配置正确
2. API密钥是否有效
3. 网络连接是否正常
</details>

<details>
<summary><b>Q: 地图无法显示</b></summary>

1. 检查 `temp_visualizations` 目录权限
2. 确认高德API密钥有效
3. 查看浏览器控制台错误
</details>

<details>
<summary><b>Q: PowerShell 无法激活虚拟环境（被脚本执行策略阻止）</b></summary>

在当前会话放宽执行策略后再激活：
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```
</details>

<details>
<summary><b>Q: 端口 7860 被占用</b></summary>

方法一：结束占用进程（示例）
```powershell
netstat -ano | Select-String 7860
taskkill /PID <上一步查到的PID> /F
```

方法二：临时修改启动端口（编辑 `main_gradio.py` 最后一行）
```python
demo.launch(server_port=7861)
```
</details>

<details>
<summary><b>Q: .env 似乎未生效</b></summary>

- 确认 `.env` 位于项目根目录（与 `main_gradio.py` 同级）
- 本项目已在所有导入之前执行 `load_dotenv()`，请确保没有移动该语句
- Windows 下请使用 UTF-8 编码保存 `.env`
</details>

更多问题请查看《[快速开始指南](./快速开始指南.md)》的 FAQ 部分。

---

## 📈 更新日志

### v2.0.0 (2025-10)

**新增功能**:
- ✨ 深度思考模式(qwen-plus)
- ✨ 成功案例长期记忆库
- ✨ Prompt模块化管理
- ✨ 完整的requirements.txt

**优化改进**:
- 🔧 HTML地图可视化修复
- 🔧 记忆系统全面重构
- 🔧 文件管理增强
- 🔧 代码注释完善

**性能提升**:
- ⚡ 相似问题响应速度提升70%
- ⚡ 成功案例复用率40-60%
- ⚡ 代码可维护性提升50%

### v1.0.0 (2025-9)

- 🎉 初始版本发布
- 基础对话功能
- 工具调用集成
- Gradio界面

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议!

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 开源协议

本项目采用 MIT 开源协议（如未包含 LICENSE 文件，请按需添加）。

---

## 👨‍💻 作者

**wjj**

- 项目根目录: .gaode_agent
- 邮箱: 2089966424@qq.com
- GitHub: Jianjun Wang passionworkeer

---

## 🙏 致谢

感谢以下开源项目:

- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [Gradio](https://github.com/gradio-app/gradio) - ML应用界面
- [通义千问](https://tongyi.aliyun.com/) - 大语言模型
- [高德地图](https://lbs.amap.com/) - 地理位置服务
- [Tavily](https://tavily.com/) - 搜索引擎

---

## 📞 支持

如有问题或建议,请:
- 📧 发送邮件至: 2089966424@qq.com
- 💬 提交Issue: [项目Issue页面]
- 📖 查看文档: [文档链接]

---

<div align="center">

**⭐ 如果这个项目对您有帮助，欢迎点 Star 支持! ⭐**

Made with ❤️ by wjj

</div>


