import streamlit as st
from news_fetch import NewsArticle
from chatbot import ConversationalQA

# %%
st.set_page_config(page_title="SMART News Reader - Chatbot", page_icon="📈")

st.title('Smart News Reader App - Chatbot')
with st.expander("🌟 Welcome to Smart News Reader!", expanded=False):
    st.write(
        """     
    - Welcome to the Smart News Reader app, your gateway to personalized news exploration!
    - Uncover key insights, get concise summaries, and engage in insightful Q&A about your favorite articles.
    - Use the intuitive controls on the left to tailor your news-reading experience to your preferences.
        """
    )
# Step-by-Step Guide
with st.expander("🚀 How to Use", expanded=False):
    st.write(
        """
    1. **Paste Article URL:**
        - Copy the URL of the news article you want to explore.
        - Paste it into the 'Enter the News Article URL' text box on the left sidebar.

    2. **Choose Features:**
        - Toggle the switches on the left sidebar to enable or disable features.
        - Set the minimum and maximum n-gram values for keyword extraction (if enabled).

    3. **Explore Keywords:**
        - If enabled, discover key words extracted from the article based on your chosen n-gram range.

    5. **Summarize Content:**
        - If enabled, find a concise summary of the article's content.

    6. **Ask Questions:**
        - If enabled, interact with the Question Answering feature to get answers related to the article.
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
    if 'chat_history' in st.session_state:
        st.session_state.chat_history = []
def on_new_url():
    st.session_state['static_run'] = False
    st.session_state.chatbot = load_chatbot()
    st.session_state.chatbot.create_vector_db(news.article.text)
    st.session_state.chatbot.create_pipe()
@st.cache_resource
def load_chatbot():
    return ConversationalQA()


with st.sidebar:
    st.text_input('Enter the News Article URL',
                  value='https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html',
                  key='url_input',
                  on_change=on_new_url()
                  )

    st.toggle(label='Show relavent document',
              value=True,
              key='show_doc')


# %%
st.divider()
st.subheader('Question Answering about Article')


question = st.chat_input("Ask me questions from the article..")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if question:
    for mem_question, mem_answer in st.session_state.chat_history:
        with st.chat_message('human'):
            st.write(mem_question)
        with st.chat_message('ai'):
            st.write(mem_answer)
    # for message in st.session_state.chatbot.memory.chat_memory.messages:
    #     with st.chat_message(message.type):
    #         st.write(message.content)

    with st.chat_message('human'):
        st.write(question)
    with st.spinner('Asking Llama-2 ...'):
        result = st.session_state.chatbot.infer(question, st.session_state.chat_history)
    with st.chat_message('ai'):
        st.write(result['answer'])

    st.session_state.chat_history.append((question, result['answer']))
    if st.session_state.show_doc:
        st.dataframe({i:doc.page_content for i,doc in enumerate(result['source_documents'])})

st.button('New Question? Clear Chat', on_click=clear_chat)