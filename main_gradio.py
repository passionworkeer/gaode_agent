import os
from dotenv import load_dotenv

# ✅ 在所有其他导入之前加载环境变量
load_dotenv()

import gradio as gr
from gradio import themes
import asyncio
import re
import logging
import os
from chat_agent_qwen.agent_self.chat_agent_qwen_3_max import QwenModel
from chat_agent_qwen.agent_memory.memory import MemoryManager
from chat_agent_qwen.agent_self.agent import Agent

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- 全局单例 ---
agent_instance = None
memory_manager = None

# 初始化Agent
async def init_agent():
    global agent_instance, memory_manager
    if agent_instance is None:
        model = QwenModel()
        memory_manager = MemoryManager()
        agent_instance = Agent(model, memory_manager)
    return agent_instance

# ----- 辅助：同步获取 agent.run（在 generator 里调用） -----
def run_agent_sync(user_input, user_id, deep_thinking_enabled=False):
    return asyncio.run(_run_agent_and_extract(user_input, user_id, deep_thinking_enabled))

async def _run_agent_and_extract(user_input, user_id, deep_thinking_enabled=False):
    ag = await init_agent()
    # ✅ 全局注入用户身份，用于 visualization_tool / file_tool 识别
    os.environ["CURRENT_USER_ID"] = str(user_id or "anonymous")
    logger.info(f"✅ 已注入用户身份: {os.environ['CURRENT_USER_ID']}")

    if deep_thinking_enabled:
        ag.model.llm.model_name = "qwen-plus"
        logger.info("🧠 已切换至深度思考模式 (qwen-plus)")
    else:
        ag.model.llm.model_name = "qwen-max"
        logger.info("⚡ 使用快速模式 (qwen-max)")
    
    response = await ag.run(user_input, user_id=user_id)

    # 检测地图或图片
    map_match = re.search(r'(\./temp_visualizations/.*?\.html)', response)
    img_match = re.search(r'(\./temp_visualizations/.*?\.(?:png|jpg|jpeg|gif))', response)

    cleaned = response
    new_map = None
    new_img = None
    
    if map_match:
        new_map = map_match.group(1)
        cleaned = cleaned.replace(new_map, "").strip()
        logger.info(f"🗺️ 检测到地图文件: {new_map}")
    
    if img_match:
        new_img = img_match.group(1)
        cleaned = cleaned.replace(new_img, "").strip()
        logger.info(f"🖼️ 检测到图片文件: {new_img}")

    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned, new_map, new_img
# ----- 异步Generator函数 -----
async def agent_chat_gen_streaming(
    user_input, 
    user_id, 
    history, 
    current_map, 
    current_images,
    deep_thinking_enabled
):
    if not user_input:
        yield "", history, history, gr.update(visible=bool(current_map)), gr.update(visible=bool(current_images)), current_map, (current_images or [])
        return

    if history is None:
        history = []

    history.append([user_input, ""]) 
    yield "", history, history, gr.update(visible=bool(current_map)), gr.update(visible=bool(current_images)), current_map, (current_images or [])

    ag = await init_agent()
    import os
    # ✅ 全局注入用户身份（核心修复点）
    os.environ["CURRENT_USER_ID"] = str(user_id or "anonymous")
    logger.info(f"✅ 已注入用户身份: {os.environ['CURRENT_USER_ID']}")

    if deep_thinking_enabled:
        ag.model.llm.model_name = "qwen-plus"
        logger.info("🧠 已切换至深度思考模式 (qwen-plus)")
    else:
        ag.model.llm.model_name = "qwen-max"
        logger.info("⚡ 使用快速模式 (qwen-max)")
    
    try:
        # 用于收集最终完整回复
        final_response = ""
        map_update = gr.update(visible=bool(current_map))
        img_update = gr.update(visible=bool(current_images))
        new_map_content = current_map
        images_list = list(current_images or [])

        # 异步流式输出
        async for chunk in ag.run(user_input, user_id=user_id):
            if chunk:
                final_response += chunk
                # 实时处理地图和图片
                import re
                file_pattern = r'((?:user_data|temp_visualizations)/[a-zA-Z0-9_./\-]+?\.(?:html|png|jpg|jpeg|gif))'
                http_img_pattern = r'(https?://[^\s)\'"<>]+?\.(?:png|jpg|jpeg|gif))'
                matches = re.findall(file_pattern, chunk)
                http_imgs = re.findall(http_img_pattern, chunk)
                cleaned_chunk = chunk
                for raw_path in matches:
                    normalized_path = raw_path.replace('\\', '/')
                    absolute_path = os.path.abspath(normalized_path)
                    if normalized_path.endswith('.html'):
                        try:
                            with open(absolute_path, "r", encoding="utf-8") as f:
                                html_content = f.read()
                            map_update = gr.update(value=html_content, visible=True)
                            new_map_content = html_content
                            logger.info(f"🗺️ 地图已加载并显示: {absolute_path}")
                            cleaned_chunk = cleaned_chunk.replace(raw_path, "").strip()
                        except Exception as e:
                            logger.error(f"读取地图文件失败: {e}")
                    elif normalized_path.endswith(('.png', '.jpg', '.jpeg', 'gif')):
                        try:
                            if absolute_path not in images_list:
                                images_list.append(absolute_path)
                            img_update = gr.update(value=images_list, visible=True)
                            logger.info(f"🖼️ 图片已加载并显示: {absolute_path}")
                            cleaned_chunk = cleaned_chunk.replace(raw_path, "").strip()
                        except Exception as e:
                            logger.error(f"加载图片失败: {e}")
                # 处理远程图片 URL
                for url in http_imgs:
                    try:
                        if url not in images_list:
                            images_list.append(url)
                            logger.info(f"🖼️ 远程图片加入展示队列: {url}")
                        img_update = gr.update(value=images_list, visible=True)
                    except Exception as e:
                        logger.error(f"加载远程图片失败: {e}")
                # 实时更新 history
                history[-1][1] += cleaned_chunk
                # 只要有新内容就 yield 到前端
                yield "", history, history, map_update, img_update, new_map_content, images_list

        # 最终清理（去除多余空白）
        cleaned_response = final_response
        cleaned_response = re.sub(r'[ \t]+', ' ', cleaned_response)
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response).strip()
        history[-1][1] = cleaned_response
        yield "", history, history, map_update, img_update, new_map_content, images_list

    except Exception as e:
        logger.error(f"Agent执行失败: {e}", exc_info=True)
        history[-1][1] = f"抱歉，处理您的请求时遇到错误: {str(e)}"
        yield "", history, history, gr.update(visible=bool(current_map)), gr.update(visible=False), current_map, (current_images or [])


# 清空上下文
def temp_clear_context(user_id):
    global memory_manager
    if memory_manager:
        try:
            memory_manager.clear_context(user_id)
            msg = [["", "✅ 上下文已清空，可以开始新的对话了。"]]
        except Exception as e:
            msg = [["", f"❌ 清空上下文时出错: {e}"]]
    else:
        msg = [["", "⚠️ MemoryManager 尚未初始化"]]
    return "", msg, msg, None, None, None, []


# --- Gradio UI ---
with gr.Blocks(
    theme=themes.Soft(),
    css="""
#chatbot { min-height: 60vh; }
.user { text-align: right; }
""",
    title="智慧旅行助手 by wjj"
) as demo:

    # --- 状态 ---
    user_id = gr.State("")
    chat_history = gr.State([])
    current_map = gr.State()
    current_images = gr.State([])

    gr.Markdown("# 智慧旅行助手 🗺️")
    gr.Markdown("我是一个智能旅行规划助手，可以为您规划行程、查询信息，并生成交互式地图。")

    chatbot = gr.Chatbot(label="聊天窗口", elem_id="chatbot", bubble_full_width=False)

    with gr.Row():
        map_display = gr.HTML(label="地图展示", visible=False, elem_id="map_display")
        img_display = gr.Gallery(label="图片预览", visible=False, columns=3, height=200, preview=True, allow_preview=True, elem_id="img_display")

    with gr.Row():
        deep_thinking_toggle = gr.Checkbox(
            label="🧠 启用深度思考模式 (qwen-plus)", 
            value=True,
            info="开启后使用更强的模型进行深度推理，关闭则使用快速响应模型。"
        )

    with gr.Row():
        user_input = gr.Textbox(placeholder="请输入您的出行需求...", lines=1, scale=8, container=False)
        send_btn = gr.Button("发送", scale=1, variant="primary")
        clear_btn = gr.Button("清空", scale=1)

    demo.load(
        None,
        None,
        user_id,
        js="""
        () => {
            let storedId = localStorage.getItem('travel_agent_user_id');
            if (!storedId) {
                storedId = 'user_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now();
                localStorage.setItem('travel_agent_user_id', storedId);
            }
            return storedId;
        }
        """
    )

    click_params = {
        "fn": agent_chat_gen_streaming,
        "inputs": [user_input, user_id, chat_history, current_map, current_images, deep_thinking_toggle],
        "outputs": [user_input, chatbot, chat_history, map_display, img_display, current_map, current_images],
        "queue": True
    }

    send_btn.click(**click_params).then(lambda: gr.update(value=None), None, [user_input])
    user_input.submit(**click_params).then(lambda: gr.update(value=None), None, [user_input])

    clear_btn.click(
        fn=temp_clear_context,
        inputs=[user_id],
        outputs=[user_input, chatbot, chat_history, map_display, img_display, current_map, current_images]
    )

    gr.Markdown("---\n*Powered by wjj. 仅用于技术演示。*")

if __name__ == "__main__":
    demo.launch()
