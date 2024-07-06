import streamlit as st
import pickle
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Loading embedding model from model.pkl
embeddings_model = pickle.load(open('model.pkl', 'rb'))

# Connecting to existing database
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

# Creating prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context: {question}
"""

# Function for generating prompts
def query_result(query):
    results = db.similarity_search_with_relevance_scores(query, k=10)
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    return prompt

# Loading llama2 llm model
llm = Ollama(model="llama2")

if __name__ == '__main__':
    st.title('CHATBOT')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Displaying the chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for user query
    query = st.chat_input("Type your message here...")

    if query:
        # Generate prompt and get response from LLM
        prompt = query_result(query)
        response_text = llm.invoke(prompt)

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
