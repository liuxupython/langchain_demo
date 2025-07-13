import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()


model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
)


def invoke_messages(query: str):

    messages = [
        SystemMessage("你是一个专业的心理医生，请帮咨询者解答身心疑惑。"),
        HumanMessage(query),
    ]

    response = model.invoke(messages)
    output = f"""
        content:\n{response.content}\n
        response_metadata:\n{response.response_metadata}\n
        response.additional_kwargs:\n{response.additional_kwargs}\n
        response.type:\n{response.type}\n
    """.replace(
        "        ", ""
    )
    print(output)


def invoke_prompt(query: str):
    # system_template = '你是一个专业的心理医生，请帮咨询者解答身心疑惑。用户的问题如下:\n {question}'
    # prompt_template = ChatPromptTemplate.from_template(system_template)
    # prompt = prompt_template.invoke({'question': query})

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage("你是一个专业的心理医生，请帮咨询者解答身心疑惑。"),
            HumanMessage(query),
        ]
    )
    prompt = prompt_template.invoke({})
    result = model.invoke(prompt)
    print(result.content)


def stream_output(query: str):
    system_template = (
        "你是一个专业的心理医生，请帮咨询者解答身心疑惑。用户的问题如下:\n {question}"
    )
    prompt_template = ChatPromptTemplate.from_template(system_template)

    # print(prompt_template.format_messages(question=query))

    prompt = prompt_template.invoke({"question": query})

    print(prompt)

    for chunk in model.stream(prompt):
        print(chunk.text(), end="", flush=True)


if __name__ == "__main__":
    # invoke_messages('我老是担心身体不够健康，这样的想法是不是不正常？')
    invoke_prompt("我老是担心身体不够健康，这样的想法是不是不正常？")
    # stream_output('我老是担心身体不够健康，这样的想法是不是不正常？')
