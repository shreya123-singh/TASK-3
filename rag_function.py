import os
from mistralai import Mistral, SDKError
from dotenv import load_dotenv
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from json import dumps, loads
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

mistral_api = os.getenv("mistral_ai")
huggingface_api = os.getenv("huggingface")
gemini_api = os.getenv("gemini_api")
api_key= os.getenv("api_key")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True, google_api_key="AIzaSyA0Iq3ov_jPlzSxI1G3cS3X7Dm2vPRvXZM")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=gemini_api)
db_path = r"C:\Users\sumit\Desktop\my work\rag\db_faiss"
folder_path=r"C:\Users\sumit\Desktop\my work\rag\data"

def process_pdf_OCR_and_create_docs(temp_file_path, api_key,chunk_size=4049, chunk_overlap=500):
    
    # 1.temp file 
    if hasattr(temp_file_path, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(temp_file_path.read())
            temp_file_path = tmp_file.name

    #2.check file path
    if os.path.isfile(temp_file_path):
        pdf_folder = os.path.dirname(temp_file_path)
        pdf_to_process = [os.path.basename(temp_file_path)]
    elif os.path.isdir(temp_file_path):
        pdf_folder = temp_file_path
        pdf_to_process = [f for f in os.listdir(temp_file_path) if f.lower().endswith('.pdf')]
    else:
        print(f"Error: {temp_file_path} is not a valid file or directory")
        return {}

    if not pdf_to_process:
        print("Error: No PDF files found to process")
        return {}

    # 3. Initialize Mistral client to extract text from PDF
    client = Mistral(api_key=api_key)
    ocr_results = {}
    for filename in pdf_to_process:
        file_path = os.path.join(pdf_folder, filename)
        try:
            with open(file_path, "rb") as file_data:
                uploaded_pdf = client.files.upload(
                    file={
                        "file_name": filename,
                        "content": file_data,
                    },
                    purpose="ocr"
                )

            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url}
            )

            ocr_results[filename] = ocr_response

        except SDKError as e:
            print(f"Error processing {filename}: {e}")

    #4. Combine extracted text from all files
    combined_text = ""
    for filename, result in ocr_results.items():
        extracted_text = ""

        if hasattr(result, "pages"):
            extracted_text = "\n".join([
                getattr(page, "markdown", "") or getattr(page, "text", "")
                for page in result.pages
            ])
        else:
            print(f"[WARN] Unrecognized format for {filename}")
            continue

        if extracted_text.strip():
            combined_text += extracted_text.strip() + "\n"
        else:
            print(f"[WARN] No text extracted from: {filename}")

    if not combined_text.strip():
        print("[ERROR] No text was extracted from any file.")
        return []

    #5. Split the combined text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(combined_text)
    docs = [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

    return docs

#6. Store and load FAISS index
def store_and_load_faiss_index(docs, embeddings, k=20):
    try:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(db_path)

        # Load for retrieval
        faiss_index = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever
    except IndexError as e:
        raise IndexError(f"Embedding list is empty. Ensure that embeddings are generated correctly. Details: {str(e)}")

#7. Get unique data    
def get_unique_data(documents: list[list]):
    # Convert Document objects to dictionaries before serialization
    flatten_docs = [dumps(docs.dict()) for sublist in documents for docs in sublist]
    unique_docs = list(set(flatten_docs))
    # Convert dictionaries back to Document objects after deserialization
    return [Document.parse_obj(loads(docs)) for docs in unique_docs]

#8. Build query generator and retrieval chain
def build_query_generator_and_retrieval_chain(llm, retriever):
    query_generation_template = """
    You are an intelligent assistant designed to generate optimized search queries from user questions and retrieve information.

    Original User Question:
    {question}

    Generate a concise and relevant search query that can be used to retrieve accurate information from a knowledge base or document store.
    """
    prompt_perspectives = ChatPromptTemplate.from_template(query_generation_template)

    generate_queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: [line for line in x.split("\n") if line.strip()])
    )

    return generate_queries | retriever.map() | get_unique_data

#9. Build RAG chain for QA
def build_rag_chain_for_QA(question, llm,temp_file_path):
    docs=process_pdf_OCR_and_create_docs(temp_file_path, api_key,4049,500)
    retriever = store_and_load_faiss_index(docs,embeddings,20)
    query_chain  = build_query_generator_and_retrieval_chain(llm, retriever)

    # Prompt 3: Final Answer Generation
    answer_generation_template = """
    You are an AI language model assistant. Your task is to generate the response according to the user's question.

    Here is your Question: {question}
    Here is your Context to answer the question: {context}

    # Instructions:
    - Do not hallucinate.
    - If the answer is not present in the context, say "The answer is not available in the given context."
    - Follow the instruction, if any, provided by user in the question.
    """
    answer_prompt = ChatPromptTemplate.from_template(answer_generation_template)


    rag_chain = (
        {
            "context": query_chain,
            "question": itemgetter("question")
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"question":question})

#10. Build RAG chain for Summarization
def build_summarization_chain(llm,temp_file_path):
    docs=process_pdf_OCR_and_create_docs(temp_file_path, api_key,4049,500)
    retriever = store_and_load_faiss_index(docs,embeddings,20)
    query_chain  = build_query_generator_and_retrieval_chain(llm, retriever)

    # Prompt 3: Final Answer Generation
    summarization_template = """
    You are an AI assistant specializing in concise and informative summaries.

    Here is the context you need to summarize:
    {context}

    # Instructions:
    - Provide a brief and clear summary.
    - Keep it under 100 words.
    - Maintain the core information without adding extra details.
    """
    summary_prompt = ChatPromptTemplate.from_template(summarization_template)


    rag_chain = (
        {
            "context": query_chain,
        }
        | summary_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke("Give me summary")
