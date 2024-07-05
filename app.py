import streamlit as st
import pickle
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

#loading embedding model from model.pkl
embeddings_model = pickle.load(open('model.pkl', 'rb'))
#connecting to existing database
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH,embedding_function=embeddings_model)
#creating prompt template
PROMPT_TEMPLATE ="""
Answer the questio based only on the following context:
{context}

---
Answer the question based on the above context:{question}
"""

#function for generating prompts
def query_result(query):
    results  = db.similarity_search_with_relevance_scores(query,k=10)
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text,question=query)
    return prompt


#loading llama2 llm model
llm = Ollama(model="llama2")


if __name__ == '__main__':
    st.title('CHATBOT')
    query = st.text_area("enter your query", "",height=10)
    button = st.button("ask")
    if button:
        prompt=query_result(query)
        response_text = llm.invoke(prompt)
        st.write(response_text)
        
    
    
    
    