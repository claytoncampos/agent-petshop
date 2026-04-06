import os
from dotenv import load_dotenv
from rich import print
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate

load_dotenv()
llm = ChatGroq(model= os.getenv('GROQ_MODEL'), temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Voçê é um assistente do PetShop Animalia. Responda de forma educada."
    "Pergunta: {question}"
)

def invoke_llm(text):
    chain = prompt | llm
    result =  chain.invoke({"question": text})
    return result.content


if __name__ == "__main__":
    print(invoke_llm("olá LangSmith"))


