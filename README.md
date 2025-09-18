# LangChain RAG Pipeline for Chess AI Research
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline using LangChain and Hugging Face models. The system is designed to ingest and process research papers related to chess AI, allowing a user to ask domain-specific questions and receive contextually grounded answers.

The primary workflow involves loading PDF documents, splitting them into manageable chunks, generating embeddings, storing them in a Chroma vector database, and using a RetrievalQA chain to connect a retriever with a Large Language Model (LLM). The project also includes experiments to evaluate the impact of different embedding models and chunking strategies on the RAG system's performance.

## Features
Document Loading: Ingests multiple PDF documents using LangChain's PyPDFLoader.

Text Chunking: Splits documents into configurable chunks using RecursiveCharacterTextSplitter.

Vector Embeddings: Generates embeddings for text chunks using Sentence-Transformers models.

Vector Storage: Creates and persists a Chroma vector database for efficient similarity searches.

LLM Integration: Connects to an open-source model (distilgpt2) from the Hugging Face Hub.

Question-Answering: Implements a RetrievalQA chain to answer questions based on the document corpus.

Experimentation: Systematically compares the performance of:

Embedding Models: all-MiniLM-L6-v2 vs. intfloat/e5-small-v2.

Chunking Strategies: chunk_size=500/overlap=100 vs. chunk_size=300/overlap=50.

Reproducibility: Logs all environment versions, configurations, and experiment results to JSON files (env_rag.json and rag_run_config.json).

## Getting Started
Prerequisites

Python 3.8+

pip package manager

### 1. Clone the Repository

```
git clone https://github.com/jm7n7/week-4-RAG.git
cd "your-repository-directory"
```

### 2. Install Dependencies

Install the required Python packages using the provided pip command from the notebook. It's recommended to use a virtual environment.
`
pip install langchain chromadb sentence-transformers transformers langchain-community pypdf torch
`
### 3. Set Up API Keys

This project requires a Hugging Face Hub API token to download and use the LLM.

Get your token from the Hugging Face website.

If running in Google Colab, save the token as a secret named HF_TOKEN. If running locally, set it as an environment variable.

### 4. Prepare Your Documents

Place your PDF documents (e.g., maia-2.pdf, Amortized_chess.pdf) in the project directory where the notebook can access them. Update the file paths in Step 2 of the notebook to match your document names.

## How to Run
Launch the Jupyter Notebook: RAG_Hands_On.ipynb.

Ensure your HF_TOKEN is accessible.

Verify that the document paths in the "Load Your Project Documents" section are correct.

Run the notebook cells sequentially from top to bottom.

The notebook will automatically:

Load and process the PDF files.

Build the initial RAG pipeline with default settings.

Run three sample queries.

Execute the embedding and chunking experiments.

Save all configurations and results to rag_run_config.json.

## Workflow and Experiments Explained
The notebook is structured into a logical sequence of steps to build and evaluate the RAG pipeline.

Core RAG Pipeline

Load: PDF documents are loaded into memory.

Split: The documents are split into smaller text chunks. The initial configuration is a chunk size of 500 characters with an overlap of 100.

Embed & Store: The all-MiniLM-L6-v2 model generates vector embeddings for each chunk, which are then stored in a ChromaDB vector store.

Retrieve: A retriever is created to fetch the top 4 (k=4) most relevant document chunks based on a user's query.

Generate: The retrieved chunks are passed as context to the distilgpt2 LLM, which generates an answer to the query.

Experiments

To test the system's sensitivity to different components, two mini-experiments are conducted:

Embedding Swap: The entire pipeline is re-run using the intfloat/e5-small-v2 embedding model to compare its retrieval quality against the baseline all-MiniLM-L6-v2.

Chunk Sensitivity: The documents are re-chunked with a smaller size (300) and overlap (50) to analyze how chunking strategy affects the final answer quality.

The results from both experiments are saved in the rag_run_config.json file for comparison.

