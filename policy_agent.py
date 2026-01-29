import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------------------
# Embeddings (HuggingFace)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Pinecone Vector Store
# -------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------------
# LLM (Groq)
# -------------------------------
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -------------------------------
# POLICY PROMPT
# -------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a strict policy assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say:
"I cannot find this information in the policy document."

Context:
{context}

Question:
{question}
""")

# -------------------------------
# Retrieval Chain
# -------------------------------
chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------
# CLI LOOP
# -------------------------------
print("✅ Policy Agent Ready (type 'exit' to quit)\n")

while True:
    query = input("Ask Policy Question: ")
    if query.lower() == "exit":
        break

    response = chain.invoke(query)
    print("\nAnswer:", response, "\n")
