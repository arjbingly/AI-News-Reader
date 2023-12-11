# To Run: streamlit run app.py --server.port=8888
import streamlit as st
import pandas as pd

from news_fetch import NewsArticle
from zeroshot import ZeroShotClassifier
from kw_extraction import KeywordExtractor
from chatbot import ConversationalQA
from summarizer import Summarizer

from streamlit_tags import st_tags

# %%
st.set_page_config(
    page_title="SMART News Reader",
    page_icon="🎈",
)
# %%
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

    6. **Ask Questions:**
        - Switch to the Conversational QA page, interact with the Question Answering Chatbot  to get answers related to the article.
        """
    )
# %%


with st.sidebar:
    st.text_input('Enter the News Article URL',
                  value='https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html',
                  key='url_input',
                  )

    st.toggle(label='Summarization',
              value='True',
              key='do_summarization')

    st.toggle(label='Key Word Extractor',
              value='True',
              key='do_key_word')
    if st.session_state.do_key_word:
        st.slider("Minimum Ngram", min_value=1, max_value=4, value=1, step=1, key='min_Ngrams')
        st.slider("Maximum Ngram", min_value=1, max_value=4, value=1, step=1, key='max_Ngrams')
        if st.session_state.min_Ngrams > st.session_state.max_Ngrams:
            st.write(":red[Min. Ngram can't be greater than Max. Ngram]")
            st.session_state.max_Ngrams = st.session_state.min_Ngrams

    st.toggle(label='Zero-Shot Classifier',
              value='True',
              key='do_zeroshot')

# %%
# Load models
@st.cache_resource
def load_summarizer():
    return Summarizer()


@st.cache_resource
def load_chatbot():
    return ConversationalQA()


@st.cache_resource
def load_keyword_extractor(min_Ngrams, max_Ngrams):
    return KeywordExtractor(ngram_range=(min_Ngrams, max_Ngrams))


@st.cache_resource
def load_zeroshot():
    return ZeroShotClassifier()


# %%
news = NewsArticle(st.session_state.url_input)
lang = news.article.meta_lang
st.header(news.article.title)
st.subheader(' ,'.join(news.article.authors))
st.caption(f'Source: {news.article.url} ')
st.caption(f'Publish Date: {news.article.publish_date}')

st.image(news.article.top_image)

#%%
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = load_chatbot()

with st.spinner('Loading the bot...'):
    st.session_state.chatbot.create_vector_db(news.article.text)
    st.session_state.chatbot.create_pipe()

#%%
if st.session_state.do_summarization:
    st.divider()
    st.subheader('Summary')
    with st.spinner('Summarizing the article...'):
        summarizer = load_summarizer()
        summary = summarizer.summarize(news.article.text)
        st.session_state['summary'] = summary
    st.write(st.session_state.summary)
# %%
if st.session_state.do_key_word:
    st.divider()
    st.subheader('Key Words')
    with st.spinner('Extracting key words from the article...'):
        keyword_extractor = load_keyword_extractor(st.session_state.min_Ngrams, st.session_state.max_Ngrams)
        key_words_df = pd.DataFrame(keyword_extractor.extract_keywords(news.article.text),
                                    columns=['Key Word', 'Score'])
        col1, col2, col3 = st.columns([1, 2, 1])
        st.session_state['key_words_df'] = key_words_df

    col2.dataframe(st.session_state.key_words_df, hide_index=True, use_container_width=True)
# %%
if st.session_state.do_zeroshot:
    st.divider()
    st.subheader('Zero-Shot Classifier')
    keywords = st_tags(label='',
                       text='Press enter to add more',
                       value=['Politics', 'Science', 'Business', 'Travel'])
    classifier = load_zeroshot()
    with st.spinner('Zero-Shot Classifying...'):
        zeroshot_results = classifier.classify(news.article.text, keywords)
        zeroshot_df = pd.DataFrame(classifier.classify(news.article.text, keywords))
        zeroshot_df = zeroshot_df.drop(columns='sequence')
        zeroshot_df = zeroshot_df.sort_values('scores', ascending=False)
        st.session_state['zeroshot_df'] = zeroshot_df

    col1, col2, col3 = st.columns([1, 2, 1])
    col2.dataframe(st.session_state.zeroshot_df, hide_index=True, use_container_width=True)
# %%

# %%

