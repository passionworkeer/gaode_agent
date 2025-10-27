import asyncio
import os
import json
# 假设你的 TavilySearchToolManager 在 Tavilysearch_tool.py 文件中
# 请根据你的实际项目结构调整导入路径
# from Tavilysearch_tool import TavilySearchToolManager
# --- 修改导入 ---
# 为了更清晰地看到导入是否成功，可以显式导入
try:
    from Tavilysearch_tool import TavilySearchToolManager
    print("✅ 成功导入 TavilySearchToolManager")
except ImportError as e:
    print(f"❌ 导入 TavilySearchToolManager 失败: {e}")
    print("   请检查 Tavilysearch_tool.py 文件是否存在，以及 TavilySearchToolManager 类名是否正确。")
    exit(1)
# --- 导入结束 ---

async def test_tavily_image_search():
    """测试 TavilySearchTool 是否能返回图片链接"""
    print("🔍 开始测试 Tavily 图片搜索功能...")

    # --- 1. 初始化 Tavily 客户端 ---
    # 请确保你的环境变量 TAVILY_API_KEY 已设置，或者在这里直接提供 API Key
    # api_key = os.getenv("TAVILY_API_KEY") # 从环境变量获取
    api_key = "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" # 直接提供 API Key (请替换为你自己的)
    print(f"🔑 使用 API Key (前10位): {api_key[:10]}... (长度: {len(api_key)})")

    if not api_key or len(api_key) < 20: # 简单检查 Key 是否看起来合理
        print("❌ 错误: TAVILY_API_KEY 看起来不正确或为空。")
        return

    # --- 修改初始化部分 ---
    print("🔄 正在初始化 TavilySearchToolManager...")
    manager = TavilySearchToolManager(api_key=api_key)
    print(f"   Manager 实例创建成功: {type(manager)}")
    # --- 修改：正确使用 await 调用 initialize ---
    try:
        # tools = manager.initialize() # ❌ 错误：缺少 await
        tools = await manager.initialize() # ✅ 正确：使用 await
        print(f"✅ Tavily 工具初始化成功，共加载 {len(tools)} 个工具。")
        for i, tool in enumerate(tools):
             print(f"   工具 {i+1}: {type(tool).__name__} (Name: {getattr(tool, 'name', 'N/A')})")
    except Exception as e:
        print(f"❌ Tavily 工具初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    # --- 初始化结束 ---

    # --- 2. 获取 TavilySearchTool 实例 ---
    print("🔍 尝试获取 'tavily_search' 工具实例...")
    search_tool = manager.get_tool_by_name("tavily_search") # 注意工具名是 'tavily_search'
    if not search_tool:
        print("❌ 未找到 'tavily_search' 工具。")
        # 打印所有可用工具名进行调试
        available_names = [t.name for t in manager.tools] if hasattr(manager, 'tools') else []
        print(f"   可用工具名: {available_names}")
        return
    print("✅ 成功获取 'tavily_search' 工具实例。")
    print(f"   工具实例类型: {type(search_tool)}")
    print(f"   工具 name 属性: {getattr(search_tool, 'name', 'N/A')}")
    # --- 获取工具结束 ---

    # --- 3. 执行带图片的搜索 ---
    test_query = "北京故宫 高清图片" # 测试查询词
    print(f"\n🔎 执行搜索: '{test_query}'")

    try:
        print("🚀 调用 search_tool._arun ...")
        # 调用工具的 _arun 方法进行异步搜索
        # include_images=True 是关键参数
        search_result = await search_tool._arun(
            query=test_query,
            search_depth="advanced", # 使用高级搜索
            max_results=3,           # 限制结果数量
            include_images=True,     # ✅ 关键：要求返回图片
            include_answer=False     # 我们主要关心结果和图片
        )
        print("✅ 搜索执行成功。")

        # --- 4. 检查并打印结果 ---
        print("\n📄 搜索结果结构:")
        if isinstance(search_result, dict):
            print(f"  - 结果键 (Keys): {list(search_result.keys())}")

            # 检查 'results' (网页链接等)
            results_list = search_result.get('results', [])
            print(f"  - 网页结果数量: {len(results_list)}")
            if results_list:
                print(f"    - 第一个结果示例: {results_list[0] if len(results_list) > 0 else 'N/A'}")

            # 检查 'images' (原始图片列表)
            raw_images_list = search_result.get('images', [])
            print(f"  - 原始图片数量 (images): {len(raw_images_list)}")
            if raw_images_list:
                print(f"    - 前3个原始图片示例: {raw_images_list[:3] if len(raw_images_list) >= 3 else raw_images_list}")

            # 检查我们代码中添加的 'image_urls' (提取后的URL列表)
            processed_image_urls = search_result.get('image_urls', [])
            print(f"  - 提取后的图片URL数量 (image_urls): {len(processed_image_urls)}")
            if processed_image_urls:
                print("    - 提取到的图片URL:")
                for i, url in enumerate(processed_image_urls):
                    print(f"      [{i+1}]: {url}")
            else:
                print("    ⚠️  'image_urls' 列表为空。")
                if not raw_images_list:
                    print("    ⚠️  原始 'images' 列表也为空，Tavily API 可能未返回图片。")
                else:
                    print("    ⚠️  原始 'images' 列表不为空，但提取逻辑可能有问题。")
        else:
            print(f"  - 搜索返回了非字典类型: {type(search_result)}")
            print(f"  - 内容: {search_result}")

    except Exception as e:
        print(f"❌ 搜索执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # --- 修改：正确运行异步函数 ---
    # asyncio.run(test_tavily_image_search()) # ✅ 这是推荐的方式
    # 或者，如果你的环境需要更明确的事件循环管理（较少见）：
    try:
        asyncio.run(test_tavily_image_search())
    except RuntimeError:
        # 在某些 Jupyter Notebook 环境中可能需要
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_tavily_image_search())
    # --- 运行结束 ---