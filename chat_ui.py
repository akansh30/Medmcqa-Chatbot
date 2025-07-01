import streamlit as st
from chatbot.flow import build_chatbot_graph

chatbot = build_chatbot_graph()

st.set_page_config(page_title="MedMCQA Chatbot", page_icon=":)")
st.title("MedMCQA Medical Chatbot")
st.markdown("Ask a medical question. The bot will answer *only* if it finds a match in the MedMCQA dataset.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    result = chatbot.invoke({"query": user_input})
    bot_response = result["response"]
    st.session_state.messages.append({"role": "bot", "content": bot_response})


for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
