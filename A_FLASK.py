from flask import Flask, request, jsonify
import pickle
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
import google_api_key as GG  # it is my own python file in which I have saved API key

genai.configure(api_key=GG.Google_api_key)

model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config=genai.GenerationConfig(
        max_output_tokens=2000,
        temperature=0.9,
    ))

# Loading embedding model from model.pkl
embeddings_model = pickle.load(open('model.pkl', 'rb'))

# Connecting to existing database
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

# Creating prompt template
PROMPT_TEMPLATE = """
Provide an informative response based on the following information:
{context}

---
Question: {question}

Please ensure that the response is plain text, without any markdown, HTML, or other formatting.
"""


app = Flask(__name__)

def query_result(query):
    results = db.similarity_search_with_relevance_scores(query, k=10)
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    return prompt

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    try:
        prompt = query_result(user_query)
        response_text = model.generate_content(prompt)
        answer = response_text.text
        return jsonify({"response": answer})
    except ValueError:
        return jsonify({"response": "Hmm, there seems to be a problem. Can you rephrase your question?"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
