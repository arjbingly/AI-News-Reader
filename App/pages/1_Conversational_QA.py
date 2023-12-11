import streamlit as st
from news_fetch import NewsArticle
from chatbot import ConversationalQA

# %%
st.set_page_config(page_title="SMART News Reader - Chatbot", page_icon="📈")
st.session_state.update(st.session_state)
st.title('Smart News Reader App - Chatbot')

# Step-by-Step Guide
with st.expander("🚀 How to Use", expanded=False):
    st.write(
        """
    1. **Go Ahead and Ask the Chatbot Questions:**
        - The chatbot will remember previous questions, so that you can ask follow up questions.

    2. **Choose Show relevant document**
        - Toggle the switches on the left sidebar to enable or disable Show relevant document.
        - You can see the relevant part from the article the chatbot used to answer your question.
        - In case you want to fact-check
        """
    )


news = NewsArticle(st.session_state.url_input)
lang = news.article.meta_lang
st.header(news.article.title)
st.subheader(' ,'.join(news.article.authors))
st.caption(f'Source: {news.article.url} ')
st.caption(f'Publish Date: {news.article.publish_date}')
st.image(news.article.top_image)

def clear_chat():
    if 'chatbot' in st.session_state:
        st.session_state.chatbot.clear_memory()
def on_new_url():
    clear_chat()
    st.session_state.chatbot = load_chatbot()
    st.session_state.chatbot.create_vector_db(news.article.text)
    st.session_state.chatbot.create_pipe()
@st.cache_resource
def load_chatbot():
    return ConversationalQA()


with st.sidebar:
    st.toggle(label='Show relevant document',
              value=False,
              key='show_doc')


# %%
st.divider()
st.subheader('Question Answering about Article')


question = st.chat_input("Ask me questions from the article..")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if question:
    for message in st.session_state.chatbot.memory.chat_memory.messages:
        with st.chat_message(message.type):
            st.write(message.content)

    with st.chat_message('human'):
        st.write(question)
    with st.spinner('Asking Llama-2 ...'):
        result = st.session_state.chatbot.infer(question, st.session_state.chat_history)
    with st.chat_message('ai'):
        st.write(result['answer'])

    st.session_state.chat_history.append((question, result['answer']))
    if st.session_state.show_doc:
        st.table({i:doc.page_content for i,doc in enumerate(result['source_documents'])})

st.button('New Question? Clear Chat', on_click=clear_chat)