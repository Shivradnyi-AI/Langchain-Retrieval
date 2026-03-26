import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# configure API key

# initiate API key from environment variable


if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")



# prompt template

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a helpful assistant.\n"
        "You maintain context across a conversation.\n\n"
        "Conversation so far:\n"
        "{history}\n\n"
        "User: {input}\n"
        "Assistant:"
    )
)

# Memory

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

# chat model

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# chain

chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=False
)

# conversation function

def chat(user_input: str) -> str:
    return chat_chain.run(input=user_input)


if __name__ == "__main__":
    print(chat("Hi, my name is Shiv."))
    print(chat("What is my name?"))
    print(chat("Can you remind me again?"))
