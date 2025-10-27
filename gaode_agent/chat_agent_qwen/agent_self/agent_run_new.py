# 这是新版本的 run 方法，需要替换到 agent.py 中

async def run(
    self, 
    user_input: str, 
    user_id: str = "default", 
    use_icl: bool = False,
    strategy: ExecutionStrategy = ExecutionStrategy.GRACEFUL_DEGRADE,
    stream_callback: Optional[Callable[[str], Coroutine]] = None
):
    """执行Agent主流程（异步生成器，支持流式输出）
    
    Args:
        user_input: 用户输入
        user_id: 用户ID
        use_icl: 是否使用ICL
        strategy: 执行策略 (失败处理方式)
        stream_callback: 废弃参数，保留为了向后兼容
    
    Yields:
        str: 流式输出的文本片段
    """
    full_response = ""
    
    try:
        # ✅ 确保工具已异步注册（包括MCP）
        if not self.tools:
            await self._register_tools()
            logger.info(f"✅ 已注册 {len(self.tools)} 个工具")
        
        self.memory.update_history(user_id, {"role": "user", "content": user_input})
        memory = self.memory.load_memory(user_id)

        # 1. 意图分类
        intent = await self._classify_intent(user_input, memory["conversation_history"])
        chunk = f"🔍 意图分析完成: **{intent.value}**\n\n"
        full_response += chunk
        yield chunk

        # 2. 根据意图执行不同逻辑
        if intent == Intent.GENERAL_CHAT:
            # 构建用于普通聊天的 Prompt
            memory = self.memory.load_memory(user_id)
            history = memory["conversation_history"]
            
            # 使用 ICL Agent 的示例（如果启用且有示例）
            icl_examples = ""
            if use_icl and self.icl_agent.examples:
                icl_examples = "\n\n".join([f"示例 {i+1}:\n用户: {ex['query']}\n助手: {ex['response']}" for i, ex in enumerate(self.icl_agent.examples)])
            
            system_prompt = f"""你是一个智能助手，可以与用户自然对话。
        {icl_examples}
        """
            # 构建 Messages
            messages = [{"role": "system", "content": system_prompt}]
            messages = MessageValidator.safe_extend_history(messages, history, max_count=5)
            messages.append({"role": "user", "content": user_input})
            validated_messages = MessageValidator.validate_messages(messages)

            # 调用模型流式生成
            response_generator = self.model.stream_generate(validated_messages) 
            
            # 流式输出
            if hasattr(response_generator, '__aiter__'):
                async for chunk in response_generator:
                    full_response += chunk
                    yield chunk
            elif hasattr(response_generator, '__iter__'):
                for chunk in response_generator:
                    full_response += chunk
                    yield chunk
            else:
                # 如果不是生成器，直接输出
                chunk = str(response_generator)
                full_response += chunk
                yield chunk
                
            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
            return

        if intent == Intent.TOOL_INFO_QUERY:
            reply = "我具备以下能力：\n" + self.list_tools()
            full_response += reply
            yield reply
            self.memory.update_history(user_id, {"role": "assistant", "content": reply})
            return

        if intent == Intent.KNOWLEDGE_QUERY_ICL:
            chunk = "好的，我将使用我的知识库为您快速解答...\n\n"
            full_response += chunk
            yield chunk
            tool = self.tools["in_context_learning_search"]
            result = await tool.arun(query=user_input)
            result_str = str(result)
            full_response += result_str
            yield result_str
            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
            return

        if intent == Intent.KNOWLEDGE_QUERY_RAG:
            chunk = "正在查询本地知识库...\n\n"
            full_response += chunk
            yield chunk
            tool = self.tools["rag_query"]
            result = await tool.arun(query=user_input)
            result_str = str(result)
            full_response += result_str
            yield result_str
            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
            return

        # --- 默认执行复杂任务逻辑 ---
        chunk = "好的，请稍等，我正在思考如何处理您的请求...\n\n"
        full_response += chunk
        yield chunk

        steps = await self.plan_tasks(user_input, user_id)
        if not steps:
            reply = "抱歉，我无法为您的请求规划出有效的执行步骤。请尝试换一种方式提问，或者描述得更具体一些。"
            full_response += reply
            yield reply
            self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
            return

        chunk = "我已经制定了如下计划：\n"
        full_response += chunk
        yield chunk
        for i, step in enumerate(steps):
            chunk = f"   - 步骤 {i+1}: {step.goal}\n"
            full_response += chunk
            yield chunk
        chunk = "\n现在，我将开始执行这些步骤...\n\n"
        full_response += chunk
        yield chunk

        step_context = StepContext()
        steps_results = []
        
        for i, step in enumerate(steps):
            chunk = f"**正在执行步骤 {i+1}: {step.goal}**\n"
            full_response += chunk
            yield chunk
            try:
                resolved_params = step_context.replace_placeholders(step.parameters, i)
                resolved_step = TaskStep(goal=step.goal, tool_name=step.tool_name, parameters=resolved_params)
                logger.info(f"步骤 {i}: {step.goal} | 解析后参数: {resolved_params}")
                chunk = f"   - 调用工具: `{step.tool_name}`\n"
                full_response += chunk
                yield chunk
                chunk = f"   - 提供参数: `{json.dumps(resolved_params, ensure_ascii=False, indent=2)}`\n"
                full_response += chunk
                yield chunk
            except ValueError as e:
                error_msg = f"参数解析失败: {e}"
                logger.error(f"步骤 {i} {error_msg}")
                chunk = f"   - ❌ **错误**: {error_msg}\n"
                full_response += chunk
                yield chunk
                if strategy == ExecutionStrategy.FAIL_FAST:
                    final_reply = f"抱歉，任务在'{step.goal}'步骤中断，因为参数准备失败。"
                    full_response += f"\n{final_reply}"
                    yield f"\n{final_reply}"
                    self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                    return
                success, result = False, {"error": error_msg}
                resolved_step = step
            else:
                success, result = await self.execute_step(resolved_step, user_id)
            
            if not success:
                error_msg = f"步骤 {i+1} 执行失败: {result}"
                logger.error(error_msg)
                chunk = f"   - ❌ **错误**: {result}\n\n"
                full_response += chunk
                yield chunk
                if strategy == ExecutionStrategy.FAIL_FAST:
                    reply = f"抱歉，在执行'{step.goal}'时遇到问题：{result}\n\n请尝试重新描述您的需求。"
                    full_response += reply
                    yield reply
                    self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
                    return
                elif strategy == ExecutionStrategy.GRACEFUL_DEGRADE:
                    logger.info(f"步骤失败，降级为纯对话模式")
                    chunk = "哎呀，执行计划遇到了一点小问题。我将尝试根据现有信息为您总结回答。\n"
                    full_response += chunk
                    yield chunk
                    break
            
            try:
                if isinstance(result, str):
                    result_dict = RobustJSONParser.parse(result)
                    step_context.set_result(i, result_dict if result_dict and isinstance(result_dict, dict) else {"raw": result, "success": success})
                else:
                    step_context.set_result(i, result if isinstance(result, dict) else {"raw": str(result), "success": success})
                logger.info(f"✅ 步骤 {i} 结果已保存")
                chunk = f"   - ✅ **成功**: 步骤完成，结果已保存。\n\n"
                full_response += chunk
                yield chunk
            except Exception as parse_error:
                logger.warning(f"⚠️ 步骤 {i} 结果保存失败: {parse_error}, 保存原始值")
                step_context.set_result(i, {"raw": str(result), "success": success, "error": str(parse_error)})
                chunk = f"   - ⚠️ **警告**: 步骤结果保存时遇到问题: {parse_error}\n\n"
                full_response += chunk
                yield chunk

            steps_results.append((resolved_step, result))
            self.memory.update_history(user_id, {"role": "system", "content": f"执行步骤 {i}: {step.goal}\n工具: {step.tool_name}\n结果: {json.dumps(result, ensure_ascii=False, indent=2)}"})

        chunk = "所有步骤执行完毕，现在我将为您整合最终结果...\n\n---\n\n"
        full_response += chunk
        yield chunk
        
        # 调用结果整合方法（异步生成器）
        async for chunk in self.integrate_results_stream(user_input, steps_results, user_id):
            full_response += chunk
            yield chunk

        self.memory.update_history(user_id, {"role": "assistant", "content": full_response})

    except Exception as e:
        logger.error(f"Agent 主流程发生意外错误: {e}", exc_info=True)
        error_message = f"\n\n--- \n**系统错误** \n抱歉，我在处理您的请求时遇到了一个意外的问题: `{str(e)}` \n请稍后再试或联系技术支持。"
        full_response += error_message
        yield error_message
        # 确保即使在顶层异常中，最终的错误信息也被记录
        self.memory.update_history(user_id, {"role": "assistant", "content": full_response})
