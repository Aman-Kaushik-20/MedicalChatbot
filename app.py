from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from scipy.stats import fisher_exact
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

load_dotenv()

embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
from pinecone import Pinecone
api_key = os.environ.get("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=api_key)

import time
index_name = "medicalchatbot"
index = pc.Index(index_name)
# wait a moment for connection
time.sleep(1)


def retreive_from_dbs(query, k):
     query_vector=embeddings.embed_query(query)
     index_name = "medicalchatbot"
     index = pc.Index(index_name)
     results = index.query(
     namespace="real",  # replace with your actual namespace
     vector=query_vector,  # the query vector for similarity search
     top_k=k, 
     include_metadata=True  # number of top results to return
     )
     docs=""
     for result in results['matches']:     
          docs+=(result['metadata']['text'])
      
     return docs


TEMPLATE="""
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


model_path = "D:\Medical Chatbot GenAI Project_\model\llama-2-7b-chat.ggmlv3.q4_0 (1).bin"

from ctransformers.langchain import CTransformers

config = {'max_new_tokens':512, 'temperature':0.8}

import sentence_transformers

llm = CTransformers(model_file=model_path, model='TheBloke/Llama-2-7B-Chat-GGML', local_files_only=True, config=config)


quiz_chain=LLMChain(llm=llm, prompt=medical_help_prompt, output_key='answer', verbose=True)

def medical_answer(query):
     print("Entering Database")
     docs = retreive_from_dbs(query, 2)
     print(" Database Exited")
     print(f"Similar databases:{docs}")
     final_result = quiz_chain.run(
     {
         "context":docs,
         "question": query
     }
     )
     return final_result
   

@app.route("/")
def index():
      return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
      msg = request.form["msg"]
      input = msg
      print(input)
      result=medical_answer( input)
      print("Response : ", result)
      return str(result)

print("Reached the End, starting running")

if __name__ == '__main__':
     app.run(host="0.0.0.0", port= 8000, debug= True)
