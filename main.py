import os
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from rag import setup

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_api_key_openai", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = openai_api_key


st.title("LangChain x StreamLit")


@st.cache_resource
def building_chains_and_embeddings():
    return setup()
chain, rag_chain = building_chains_and_embeddings()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a StreamLit chatbot. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Message StreamLit Bot"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if "OPENAI_API_KEY" not in os.environ:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)

        rag_context = rag_chain.invoke(prompt)
        history = [HumanMessage(content=i["content"]+"\n") if i["role"] == "user" else AIMessage(
            content=i["content"]+"\n") for i in st.session_state.messages]

        response = chain.invoke({"rag_context": rag_context, "chat_history": history}, config={"callbacks":[ConsoleCallbackHandler()]})

        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
