import os
from dotenv import load_dotenv
from rich import print
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from indexar import embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs = {"k": 4})

llm = ChatGroq(model= os.getenv('GROQ_MODEL'), temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Voçê é um assistente do PetShop Animalia. Responda de forma educada.\n"
    "Contexto: {context}\nPergunta: {question}"
)

def format_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)


def invoke_llm(text):
    chain = { "context": retriever | format_chunks, "question": RunnablePassthrough()} | prompt | llm
    result =  chain.invoke(text)
    return result.content


if __name__ == "__main__":
    print(invoke_llm("quais produtos tem ?"))


