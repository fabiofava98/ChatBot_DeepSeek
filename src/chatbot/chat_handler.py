import chainlit as cl
from pipeline.rag_pipeline import RAGPipeline
from config.settings import DOCUMENTS_FOLDER
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


rag = RAGPipeline()
rag.add_documents_from_folder(DOCUMENTS_FOLDER)
qa_chain = rag.get_qa_chain()

@cl.on_chat_start
async def start():
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the DeepSeek ChatBot. How can I assist you today?"
    await msg.update()
    cl.user_session.set("chain", qa_chain)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("chain")

    if not runnable:
        await cl.Message(content="Error: The chatbot is not initialized.").send()
        return

    msg = cl.Message(content="Processing...")
    
    async for chunk in runnable.astream({"query": message.content}, config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])):
        await msg.stream_token(chunk.get("result", ""))

    await msg.send()
