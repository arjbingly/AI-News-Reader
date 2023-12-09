# To Run: streamlit run app.py --server.port=8888
import streamlit as st
import pandas as pd

from news_fetch import NewsArticle
from summarizer_beta import Summarizer
from kw_extraction import KeywordExtractor
from QnA_beta import QuestionAnswering

#%%
st.set_page_config(
    page_title="SMART News Reader",
    page_icon="🎈",
)
#%%
st.title('Smart News Reader App')
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
#%%
with st.sidebar:
    url_input = st.text_input('Enter the News Article URL',
                  )
    do_summarization = st.toggle(label='Summarization',
                            value='True')

    do_key_word = st.toggle(label='Key Word Extractor',
                            value = 'True')
    if do_key_word:
        min_Ngrams = st.slider("Minimum Ngram", min_value=1, max_value=4, value=1, step = 1)
        max_Ngrams = st.slider("Maximum Ngram", min_value=1, max_value=4, value=1, step = 1)
        if min_Ngrams > max_Ngrams:
            st.write(":red[Min. Ngram can't be greater than Max. Ngram]")
            max_Ngrams = min_Ngrams

    do_qna = st.toggle(label='Question Answering',
                            value='True')

if not url_input:
    url_input = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'

#%%
# Load models
@st.cache_resource
def load_summarizer():
    return Summarizer()

@st.cache_resource
def load_qna():
    return QuestionAnswering()

@st.cache_resource
def load_keyword_extractor(min_Ngrams, max_Ngrams):
    return KeywordExtractor(ngram_range=(min_Ngrams, max_Ngrams))

#%%
news = NewsArticle(url_input)
lang = news.article.meta_lang
st.header(news.article.title)
st.subheader(' ,'.join(news.article.authors))
st.caption(f'Source: {news.article.url} ')
st.caption(f'Publish Date: {news.article.publish_date}')
st.image(news.article.top_image)
#%%
if do_key_word:
    st.divider()
    st.subheader('Key Words')
    keyword_extractor = load_keyword_extractor(min_Ngrams, max_Ngrams)
    st.table(keyword_extractor.extract_keywords(news.article.text))
#%%
if do_summarization:
    st.divider()
    summarizer = load_summarizer()
    st.subheader('Summary')
    summary = summarizer.summarize(news.article.text)
    st.write(summary)
    # st.write(news.article.text)
#%%
if do_qna:
    st.divider()
    qna = load_qna()
    qna.create_vector_db(news.article.text)
    st.subheader('Question Answering about Article')
    question = st.chat_input("Ask me questions from the article..")
    if question:
        with st.chat_message('user'):
            # st.write(f':green[*USER* ---> {question}]')
            st.write(question)
        answers = pd.DataFrame(qna.infer(question)).sort_values('score').reset_index()
        with st.chat_message('ai'):
            # st.write(f':blue[*ANSWER* --> {qna.infer(question, news.article.text)}]')
            st.write(answers.iloc[0].answer)
    # st.write(news.article.text)