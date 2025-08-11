from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
import os
from rag_function import build_rag_chain_for_QA,build_summarization_chain

load_dotenv()

mistral_api = os.getenv("mistral_ai")
huggingface_api = os.getenv("huggingface")
gemini_api = os.getenv("gemini_api")
api_key= os.getenv("api_key")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True, google_api_key="AIzaSyA0Iq3ov_jPlzSxI1G3cS3X7Dm2vPRvXZM")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=gemini_api)
db_path = r"C:\Users\sumit\Desktop\my work\rag\db_faiss"
folder_path=r"C:\Users\sumit\Desktop\my work\rag\data"

st.image(r"C:\Users\sumit\Desktop\my work\rag\logo.png")
st.title("🤖 QA & Summarization Chatbot")

try:
    selector = st.selectbox("Select an option", ["","Summary", "QA"])
    if selector == "QA":
        st.subheader("Question Answering")
        st.write("🤖 This is a question-answering chatbot that can answer questions based on the content of a PDF file.")
        if selector:
            pdf_path = st.file_uploader("Upload your PDF file")
            question = st.text_input("Enter your question")
            if st.button("Submit"):
                if question and pdf_path:
                    result = build_rag_chain_for_QA(question, llm, pdf_path)
                    st.markdown(result)
    elif selector == "Summary":
        st.subheader("Summarization")
        st.write("📄 This is a summarization chatbot that can summarize the content of a PDF file.")
        if selector:
            pdf_path = st.file_uploader("Upload your PDF file")
            if st.button("Submit"):
                if pdf_path:
                    result=build_summarization_chain(llm, pdf_path)
                    st.markdown(result)
    else:
        st.write("🙋‍♂️ Please select an option to proceed.")
except:
    st.warning("Please select an appropriate option to proceed.")
