import time

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage

dotenv.load_dotenv()



@tool
def add(a: int, b: int) -> int:
    """计算a和b的加和"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算a和b的乘积"""
    return  a* b


tools = [add, multiply]

model = init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
)
llm_with_tools = model.bind_tools(tools)


query = '帮我计算3和4的加和，然后计算3和4的乘积'

messages: list[BaseMessage] = [HumanMessage(content=query)]

print('start： llm_with_tools.invoke(messages)')
ai_msg = llm_with_tools.invoke(messages)
print('end： llm_with_tools.invoke(messages)')
messages.append(ai_msg)

print('start: tool.invoke(tool_call)')
for tool_call in ai_msg.tool_calls:
    tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = tool.invoke(tool_call)
    messages.append(tool_msg)
print('end: tool.invoke(tool_call)')



for m in messages:
    print(m)


print('start: llm_with_tools.invoke(messages)')
result = llm_with_tools.invoke(messages)
print('end: llm_with_tools.invoke(messages)')


print('result')
print(result)
