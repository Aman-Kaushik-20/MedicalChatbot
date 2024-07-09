# Medical Chatbot GenAI Project

![Screenshot 2024-07-09 164048](https://github.com/Aman-Kaushik-20/MedicalChatbot/assets/143441723/35368e99-faa5-4c9a-bfd5-a89fce9d052a)
 
 This project is a Flask-based web application designed to provide medical assistance using an AI model. The application uses embeddings from Hugging Face and integrates with Pinecone for similarity search. It utilizes Langchain and CTransformers for language model handling and prompt management.

 # Video Tutorial
Link - https://drive.google.com/file/d/1-0iV3KRrQVijQ6S59LHSdYPPTYnQj3Jp/view?usp=sharing

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Environment Variables](#environment-variables)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Dependencies](#dependencies)
7. [License](#license)

## Features

- **Flask Web Framework**: For serving web pages and handling API requests.
- **Hugging Face Embeddings**: Used for embedding queries.
- **Pinecone Integration**: For similarity search and retrieval from the database.
- **Langchain**: For handling prompt templates.
- **CTransformers**: For local language model inference.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/medical-chatbot-genai.git
   cd medical-chatbot-genai
   ```

2. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up Pinecone**:
   - Create an account on Pinecone and get your API key.
   - Set up your Pinecone index and namespace.

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add the required environment variables (see below).

## Environment Variables

Create a `.env` file in the root directory and add the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
```

## Usage

1. **Run the Flask application**:
   ```sh
   python app.py
   ```

2. **Access the application**:
   - Open your browser and go to `http://localhost:8000`.

## File Structure

```
medical-chatbot-genai/
│
├── src/
│   └── helper.py            # Contains the function to download Hugging Face embeddings
│
├── templates/
│   └── chat.html            # HTML template for the chat interface
│
├── .env                     # Environment variables file
├── app.py                   # Main application file
├── requirements.txt         # List of Python packages required
└── README.md                # This README file
```

## Dependencies

- **Flask**: Web framework
- **Hugging Face Embeddings**: For embedding queries
- **Pinecone**: Vector database for similarity search
- **Langchain**: For handling prompt templates
- **CTransformers**: For local language model inference
- **SentenceTransformers**: For sentence embedding

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

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

### Initializing Flask Application

```python
app = Flask(__name__)
load_dotenv()
```

### Downloading Embeddings

```python
embeddings = download_hugging_face_embeddings()
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

### Flask Routes

```python
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = medical_answer(input)
    print("Response: ", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
```

This README file includes all the necessary details to understand, install, and run the project effectively.
