import dotenv
from typing import Optional, List
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

dotenv.load_dotenv()

model = init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
    temperature=0.2,
)


class Classification(BaseModel):
    sentiment: str = Field(description='文本的情感', enum=['开心', '平常心', '伤心'])
    aggressiveness: int = Field(description='文本的攻击性，从1到5，数组越大攻击性越大', enum=[1, 2, 3, 4, 5])
    language: str = Field(description='文本使用的语言', enum=['中文', '英文', 'French', '日本语'])


def analysis_information(text: str):
    system_prompt = """
    Extract the desired information from the following passage.
    Only extract the properties mentioned in the 'Classification' function.
    Passage:
    {input}
    """
    tagging_prompt = ChatPromptTemplate.from_template(system_prompt)
    prompt = tagging_prompt.format_messages(input=text)

    # 让模型结构化输出结果
    structured_llm = model.with_structured_output(Classification)

    result = structured_llm.invoke(prompt)
    print(result.model_dump())


class Person(BaseModel):
    """Information about a person."""
    name: Optional[str] = Field(description='人的名字', default=None)
    age: Optional[int] = Field(description='人的年龄', default=None)
    height_in_meters: Optional[str] = Field(description='身高(单位：米)', default=None)


class Data(BaseModel):
    """Extracted data about people."""
    people: list[Person]


def debug_node(x):
    print(f'提示词：{x}')
    return x


def extract_information(text: str):
    prompt_template = ChatPromptTemplate.from_messages(messages=[
        ('system', '从输入的文本中提取人的信息，如果你不知道请不要瞎说，直接返回None即可'),
        MessagesPlaceholder('text')
    ])
    structured_llm = model.with_structured_output(Data)
    chain = prompt_template | RunnableLambda(debug_node) | structured_llm
    result = chain.invoke({'text': [HumanMessage(text)]})
    print(result)


if __name__ == '__main__':
    # inp = "c'est la vie"
    # analysis_information(inp)

    inp = '小明上大三了，但是身高只有篮球运动员姚明身高的一半.小红今年只有15岁，但是身高已经1米7了'
    extract_information(inp)
