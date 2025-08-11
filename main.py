from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Define the QA prompt template
QA_PROMPT = PromptTemplate(
    template="""You are a helpful AI assistant tasked with answering questions based on the provided context.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If the question is not related to the context, politely point out that you can only answer questions about the given context.

Context:
{context}

Question: {question}

Provide a detailed answer that:
1. Accurate to the context
2. Well-structured
3. Easy to understand
4. Cites specific parts of the context when relevant

Answer:""",
    input_variables=["context", "question"]
)

# Initialize components
VECTOR_STORE_PATH = r"/Users/aloksingh/Documents/Finance_rag/db_faiss"
GROQ_API_KEY = "gsk_sPmovkdukPJTWGdyb3FYSLJVONejV30Fc7V6sla2AmgP"
MODEL_NAME = "mixtral-8x7b-32768"

# Initialize embeddings with updated model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load vector store
try:
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings=embedding_model,  # Pass the embedding model here
        allow_dangerous_deserialization=True
    )
    print(f"Successfully loaded vector store from {VECTOR_STORE_PATH}")
except Exception as e:
    print(f"Error loading vector store: {e}")
    raise

app = FastAPI(title="Question Answering System API")

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        # Initialize LLM
        llm = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model=MODEL_NAME
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={
                "prompt": QA_PROMPT,
                "verbose": True
            },
            return_source_documents=True
        )
        
        # Get response using invoke() instead of __call__
        result = qa_chain.invoke({"query": question.query})
        
        # Format response with source documents
        response = result["result"]
        if "source_documents" in result and result["source_documents"]:
            response += "\n\nReferences:"
            for i, doc in enumerate(result["source_documents"], 1):
                if hasattr(doc, "page_content"):
                    response += f"\n{i}. {doc.page_content[:200]}..."
        
        return {"answer": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
