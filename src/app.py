# src/app.py
import chainlit as cl
from chatbot.chat_handler import start  # Remove `src.`
from chatbot.chat_handler import on_message

if __name__ == "__main__":
    cl.run()
