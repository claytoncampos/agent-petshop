import os
from typing import TypedDict
from dotenv import load_dotenv
from rich import print
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from indexar import embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from typing_extensions import Annotated,TypedDict
from langchain_core.runnables.graph import  MermaidDrawMethod



from langgraph.graph import StateGraph,START,END
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs = {"k": 4})

llm = ChatGroq(model= os.getenv('GROQ_MODEL'), temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Voçê é um assistente do PetShop Animalia. Responda de forma educada.\n"
    "Contexto: {context}\nPergunta: {question}"
)

def format_chunks(chunks):
    texto = "\n\n".join(chunk.page_content for chunk in chunks)
    return texto[:2000]  # LIMITA

@tool
def busca_rag(query: str) -> str:
    """
    Busca de inofrmações do Pet Shop, como : 
    descrição, endereço, produtos, politicas, fidelidade e dúvidas frequentes.

    """
    chunks = retriever.invoke(query)
    return format_chunks(chunks)

class Estado(TypedDict):
    messages: Annotated[list, add_messages]





ferramentas = [busca_rag]

llm_com_ferramentas = llm.bind_tools(ferramentas)

def invoke_llm_com_ferramentas(estado: Estado)-> Estado:
    return { "messages": [llm_com_ferramentas.invoke(estado["messages"])]}


builder = StateGraph(Estado)
builder.add_node("no_llm", invoke_llm_com_ferramentas)
builder.add_node("tools", ToolNode(ferramentas))

builder.add_edge(START,"no_llm")
builder.add_conditional_edges("no_llm", tools_condition)
builder.add_edge("tools","no_llm")
builder.add_edge("no_llm", END)

graph = builder.compile()

# Gera uma imagem PNG usando o serviço online do Mermaid (não requer pygraphviz)
img_data = graph.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API
)

with open("graph.png", "wb") as f:
    f.write(img_data)

estado_global = Estado({"messages":[
        SystemMessage(content="Voçê é um assistente do PetShop Animalia. Responda de forma educada.")
        ]})

def chamar_grafo(text):
    global estado_global
    estado_global["messages"].append(HumanMessage(content=text))
    estado_global = graph.invoke(estado_global)
    return estado_global["messages"][-1].content

if __name__ == "__main__":



    print(chamar_grafo("meu nome é Clayton"))
    print(chamar_grafo("Quais produtos você tem?"))
    print(chamar_grafo("Qual o meu nome?"))



