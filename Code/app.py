import streamlit as st

from news_fetch import NewsArticle
from summarizer import Summarizer
from kw_extraction import KeywordExtractor
from QnA import QuestionAnswering

#%%
st.set_page_config(
    page_title="SMART News Reader",
    page_icon="🎈",
)
#%%
st.title('Smart News Reader App')
with st.expander("ℹ️ - About this app", expanded=True):
    st.write(
        """     
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
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

    do_qna = st.toggle(label='Question Answering',
                            value='True')

if not url_input:
    url_input = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'

news = NewsArticle(url_input)
#%%
st.header(news.article.title)
st.subheader(' ,'.join(news.article.authors))
st.caption(f'Source: {news.article.url} ')
st.caption(f'Publish Date: {news.article.publish_date}')
st.image(news.article.top_image)
#%%
if do_key_word:
    st.divider()
    st.subheader('Key Words')
    keyword_extractor = KeywordExtractor()
    st.table(keyword_extractor.extract_keywords(news.article.text))
#%%
if do_summarization:
    st.divider()
    summarizer = Summarizer()
    st.subheader('Summary')
    summary = summarizer.summarization(news.article.text)
    st.write(summary)
    # st.write(news.article.text)
#%%
if do_qna:
    st.divider()
    qna = QuestionAnswering()
    st.subheader('Question Answering')
    question = st.chat_input("Ask me questions from the article..")
    if question:
        with st.chat_message('user'):
            # st.write(f':green[*USER* ---> {question}]')
            st.write(question)
        with st.chat_message('ai'):
            # st.write(f':blue[*ANSWER* --> {qna.infer(question, news.article.text)}]')
            st.write(qna.infer(question, news.article.text))
    # st.write(news.article.text)