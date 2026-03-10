import asyncio
import logging
import json

logging.basicConfig(level=logging.DEBUG)

def test():
    from src.config import AetherForgeSettings
    from src.meta_agent import MetaAgent
    
    settings = AetherForgeSettings()
    agent = MetaAgent(settings, None, None)
    
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(agent.initialize())
    
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.modules.tunelab import tools
    
    tools_def = json.dumps(tools.get_tools(), indent=2)
    s1 = "AVAILABLE TOOLS:\\nYou have access to the following tools to act upon the tunelab subsystem:\\n"
    s2 = tools_def + "\\nCRITICAL INSTRUCTION: If the user's request matches a tool, you MUST output a JSON block containing the arguments in the format below. You may include conversational text before or after, but the JSON must be wrapped in exactly this format:\\n"
    s3 = "```json\\n" + '{"tool_name": "name_of_tool", "arguments": {"arg1": "val1"}}' + "\\n```"
    
    context = s1 + s2 + s3
    
    messages = [
        SystemMessage(content="You are the Meta-Agent." + context),
        HumanMessage(content="Trigger the OPLoRA compilation cycle right now.")
    ]
    
    print("--- GENERATING T=0.0 ---")
    response_text = agent._run_llm_sync(messages, temperature=0.0)
    print("--- RESULT T=0.0 ---")
    print(repr(response_text))
    
    print("--- GENERATING T=0.1 ---")
    response_text = agent._run_llm_sync(messages, temperature=0.1)
    print("--- RESULT T=0.1 ---")
    print(repr(response_text))

if __name__ == "__main__":
    test()
