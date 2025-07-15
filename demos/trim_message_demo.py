from langchain_core.messages import trim_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages.utils import convert_to_messages
from langchain.chat_models import init_chat_model


model = init_chat_model(
    model='deepseek-chat',
    model_provider='deepseek',
)

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("what do you call a speechless parrot"),
]

trimmer = trim_messages(
    messages=messages,
    max_tokens=100,
    token_counter=count_tokens_approximately
)


m_list = convert_to_messages(messages)

print(m_list)

