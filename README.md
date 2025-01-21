# Medical Chatbot GenAI Project


https://github.com/user-attachments/assets/f6d6621e-44f3-4009-a250-de7ef8823ed4


![image](https://github.com/user-attachments/assets/2151c4b1-d733-4f4d-929d-de1b03d082fe)


## 1. Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Variables](#environment-variables)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Dependencies](#dependencies)
7. [License](#license)

## 2. Features

- **Flask Web Framework**: For serving web pages and handling API requests.
- **Hugging Face Embeddings**: Used for embedding queries.
- **Pinecone Integration**: For similarity search and retrieval from the database.
- **Langchain**: For handling prompt templates.
- **CTransformers**: For local language model inference.



3. **Set up Pinecone**:
   - Create an account on Pinecone and get your API key.
   - Set up your Pinecone index and namespace.

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add the required environment variables (see below).

## Environment Variables
```
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

   ```
## Dependencies

- **Hugging Face Embeddings**: For embedding queries
- **Pinecone**: Vector database for similarity search
- **Langchain**: For handling prompt templates
- **CTransformers**: For local language model inference
- **SentenceTransformers**: For sentence embedding


## Detailed Code Explanation

### Importing Required Libraries

```python
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from scipy.stats import fisher_exact
from sentence_transformers import SentenceTransformer
```

### Downloading Embeddings

```python

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()
```

### Creating text chunks
```python
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks
```

### Initializing Pinecone

```python
from pinecone import Pinecone
api_key = os.environ.get("PINECONE_API_KEY")

# Configure client
pc = Pinecone(api_key=api_key)

import time
index_name = "medicalchatbot"
index = pc.Index(index_name)
# Wait a moment for connection
time.sleep(1)
```


### Creating Database
```python
from langchain_pinecone import PineconeVectorStore

index_name = "medicalchatbot"
#docs_content
docs_content = [doc.page_content for doc in text_chunks]

# Connect to the index
index = pc.Index(index_name)
embeddings

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
    namespace="namespace"
)
```

### Function to Retrieve Data from Database

```python
def retreive_from_dbs(query, k):
    query_vector = embeddings.embed_query(query)
    results = index.query(
        namespace="real",
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )
    docs = ""
    for result in results['matches']:
        docs += result['metadata']['text']
    return docs
```

### Prompt Template

```python
TEMPLATE = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

from langchain.chains import LLMChain

medical_help_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=TEMPLATE
)
```

### Loading Language Model

```python
model_path = "D:\Medical Chatbot GenAI Project_\model\llama-2-7b-chat.ggmlv3.q4_0 (1).bin"

from ctransformers.langchain import CTransformers

config = {'max_new_tokens': 512, 'temperature': 0.8}

llm = CTransformers(model_file=model_path, model='TheBloke/Llama-2-7B-Chat-GGML', local_files_only=True, config=config)
```

### Chain Setup

```python
quiz_chain = LLMChain(llm=llm, prompt=medical_help_prompt, output_key='answer', verbose=True)
```

### Function to Get Medical Answer

```python
def medical_answer(query):
    print("Entering Database")
    docs = retreive_from_dbs(query, 2)
    print("Database Exited")
    print(f"Similar databases: {docs}")
    final_result = quiz_chain.run(
        {
            "context": docs,
            "question": query
        }
    )
    return final_result
```

