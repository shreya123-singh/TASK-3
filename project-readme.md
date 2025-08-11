# RAG (Retrieval Augmented Generation) System

A modular Question Answering system built on the RAG architecture, combining document processing, vector storage, and language models to provide context-aware answers.

## System Overview

### Architecture Components

1. **Document Processing Pipeline**
   - Document ingestion (supports multiple formats)
   - Text extraction and cleaning
   - Chunking strategy
   - Embedding generation

2. **Vector Store**
   - Efficient similarity search
   - Document retrieval
   - Index management

3. **Language Model Interface**
   - Context injection
   - Response generation
   - Answer formulation

4. **API Service**
   - RESTful endpoints
   - Request handling
   - Response formatting

## Setup and Installation

### Prerequisites

```bash
# Core dependencies
python 3.8+
pip
virtualenv (recommended)

# Optional system dependencies (based on document types)
tesseract-ocr  # for image-based documents
poppler-utils  # for PDF processing
```

### Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install required packages:
```bash
pip install langchain langchain_community
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install fastapi uvicorn
pip install python-dotenv
```

3. Install document processing dependencies (as needed):
```bash
pip install pdfplumber pytesseract docx2txt
```

## Configuration

Create a `.env` file with necessary configurations:

```env
VECTOR_STORE_PATH=/path/to/vector/store
LLM_API_KEY=your_llm_api_key
EMBEDDING_MODEL_NAME=your_chosen_embedding_model
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## Usage Guide

### 1. Document Processing

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=100,
    embedding_model="your_chosen_model"
)

documents = processor.process_documents("path/to/documents")
```

### 2. Vector Store Management

```python
from vector_store import VectorStore

vector_store = VectorStore(
    embedding_model=embedding_model,
    store_path=VECTOR_STORE_PATH
)

# Index documents
vector_store.add_documents(documents)

# Save index
vector_store.save()
```

### 3. Question Answering

```python
from qa_chain import QAChain

qa_chain = QAChain(
    vector_store=vector_store,
    llm=your_chosen_llm,
    prompt_template=custom_prompt
)

response = qa_chain.answer_question("Your question here")
```

## API Reference

### Start the Server

```bash
uvicorn main:app --reload
```

### Endpoints

#### POST /ask
Submit a question to the RAG system.

Request:
```json
{
    "query": "string",
    "options": {
        "max_tokens": 500,
        "temperature": 0.7,
        "include_sources": true
    }
}
```

Response:
```json
{
    "answer": "string",
    "sources": [
        {
            "content": "string",
            "metadata": {}
        }
    ]
}
```

## Customization

### 1. Document Processors

Extend the base processor for different document types:

```python
class CustomDocumentProcessor(BaseProcessor):
    def process(self, document):
        # Custom processing logic
        pass
```

### 2. Embedding Models

Support for various embedding models:
- Hugging Face models
- OpenAI embeddings
- Custom embeddings

### 3. Vector Stores

Compatible with multiple vector stores:
- FAISS
- Chroma
- Pinecone
- Weaviate
- Custom implementations

### 4. Language Models

Support for various LLMs:
- OpenAI
- Anthropic
- Llama
- Custom models

## Best Practices

1. **Document Processing**
   - Implement proper text cleaning
   - Choose appropriate chunk sizes
   - Consider document structure

2. **Vector Storage**
   - Regular index maintenance
   - Backup strategies
   - Optimization for your use case

3. **Response Generation**
   - Well-designed prompt templates
   - Context window management
   - Source citation strategies

4. **Performance Optimization**
   - Caching strategies
   - Batch processing
   - Async operations

## Troubleshooting

Common issues and solutions:
- Vector store loading errors
- Embedding generation issues
- LLM API rate limits
- Memory management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License.
