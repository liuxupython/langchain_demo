import pathlib

import dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


# DEEPSEEK_API_KEY from env
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
)


# 通义千问-text-embedding-v3
# DASHSCOPE_API_KEY from env
embeddings = DashScopeEmbeddings(model="text-embedding-v3")


# 使用本地文件类型的向量数据库Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})


def split_pdf(filepath) -> list[Document]:
    file_path = pathlib.Path.cwd() / "创业养鸡.pdf"
    loader = PyPDFLoader(filepath)
    docs: list[Document] = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def save_pdf_to_vector_db(filepath):
    all_splits: list[Document] = split_pdf(filepath)
    ids = vector_store.add_documents(documents=all_splits)


system_template = """
请回答用户的问题，使用以下检索到的上下文来回答问题，如果你不知道请回答不知道。
context: {context}
question: {question}
"""


def invoke_chain(text):
    prompt = ChatPromptTemplate.from_template(system_template)

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model
    for chunk in chain.stream(text):
        print(chunk.text(), end="", flush=True)


def custom_invoke_chain(text):
    # 第一步：用retriever获取相关文档
    relevant_documents = retriever.invoke(text)
    context = "\n".join([doc.page_content for doc in relevant_documents])

    # 第二步：将检索结果和问题填充到prompt
    prompt = ChatPromptTemplate.from_template(system_template)
    formatted_prompt = prompt.format_messages(**{"context": context, "question": text})

    # 第三步：调用LLM生成答案
    for chunk in model.stream(formatted_prompt):
        print(chunk.text(), end="", flush=True)


if __name__ == "__main__":
    question = "主人公为什么要养鸡，他的妻子的态度是什么样的？"

    # invoke_chain(question)
    # custom_invoke_chain(question)
