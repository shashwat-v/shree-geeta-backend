from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import json

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("shreegeeta")

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={'device': 'cpu', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k":3})

# llm = ChatGroq(
#     model="mixtral-8x7b-32768",
#     api_key=groq_api_key,
#     streaming=True
# )

llm = ChatOllama(
    model="mistral",
)

template = """
You are Shree Geeta AI.

Use the Bhagavad Gita context to answer the question.

Respond in plain text. DO NOT return JSON. DO NOT use curly braces.

Format EXACTLY like this:

• Summary meaning: <1-2 line summary>

• Relevant Verses:
  - Chapter X Verse Y: <one line meaning>
  - Chapter X Verse Y: <one line meaning>
  (use 2 to 3 verses max)

• Explanation for modern practical life:
<5-10 lines practical, relatable explanation with examples in today's life>

CONTEXT:
{context}

QUESTION:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
)

if __name__ == "__main__":
    question = "How to handle stress and failure in life?"
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="")

def answer_question(question: str):
    response = rag_chain.invoke(question)
    return response.content