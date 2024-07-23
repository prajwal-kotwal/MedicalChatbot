import streamlit as st
import pickle
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
import google_api_key as GG #it is my own python file in which i have saved api key

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
"""
# Function for generating prompts
def query_result(query):
    results = db.similarity_search_with_relevance_scores(query, k=10)
    context_text = '\n\n---\n\n'.join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    
    return prompt


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
        try:
            # Generate prompt and get response from LLM
            prompt = query_result(query)
            response_text = model.generate_content(prompt)
            answer = response_text.text

            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # Display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except ValueError as e:  # Catch any ValueError
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # Display assistant message with generic error handling
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("Hmm, there seems to be a problem. Can you rephrase your question?")
            st.session_state.messages.append({"role": "assistant", "content": "Hmm, there seems to be a problem. Can you rephrase your question?"})
