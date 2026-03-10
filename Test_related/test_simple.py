import asyncio
from src.config import AetherForgeSettings
from src.meta_agent import MetaAgent
from langchain_core.messages import SystemMessage, HumanMessage

async def test():
    settings = AetherForgeSettings()
    agent = MetaAgent(settings, None, None)
    await agent.initialize()
    
    context = '''\n\nAVAILABLE TOOLS:\nYou have access to the following tools to act upon the tunelab subsystem:\n[{"name": "trigger_compilation", "description": "Trigger an immediate OPLoRA compilation cycle to learn from the Replay Buffer.", "parameters": {"type": "object", "properties": {}, "required": []}}]\n\nCRITICAL INSTRUCTION: If the user's request matches a tool, you MUST output a JSON block containing the arguments in the format below. You may include conversational text before or after, but the JSON must be wrapped in exactly this format:\n```json\n{"tool_name": "name_of_tool", "arguments": {"arg1": "val1"}}\n```\n'''
    
    messages = [
        SystemMessage(content="You are the Meta-Agent." + context),
        HumanMessage(content="Trigger the OPLoRA compilation cycle right now.")
    ]
    
    print("--- GENERATING T=0.0 ---")
    print(repr(agent._run_llm_sync(messages, temperature=0.0)))
    
    print("--- GENERATING T=0.1 ---")
    print(repr(agent._run_llm_sync(messages, temperature=0.1)))

if __name__ == "__main__":
    asyncio.run(test())
