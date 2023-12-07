from news_fetch import NewsArticle
import streamlit as st
#%%
st.title('News Reader Chatbot')

with st.sidebar:
    url_input = st.text_input('Enter the News Article URL',
                  )

if not url_input:
    url_input = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'

news = NewsArticle(url_input)

st.header(news.article.title)
st.subheader(' ,'.join(news.article.authors))
st.caption(f'Source: {news.article.url} ')
st.caption(f'Publish Date: {news.article.publish_date}')
st.image(news.article.top_image)
st.write(news.article.text)
