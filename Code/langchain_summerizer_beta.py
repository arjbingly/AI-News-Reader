from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from news_fetch import NewsArticle
from summarizer import Summarizer

# %%
url = 'https://www.bbc.com/news/world-europe-63863088'  # long news article
news = NewsArticle(url)
text = news.article.text
print(len(text))

# %%
# loader = TextLoader(text)
# loader.load()
# %%
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
#                                                chunk_overlap=50,
#                                                length_function=len,
#                                                is_separator_regex=False)
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,
                                                                          chunk_size=512,
                                                                          chunk_overlap=50)
pages = text_splitter.split_text(text)
docs = text_splitter.create_documents(pages)

# %%
summarizer = Summarizer()
summarizer.max_length = 100
#
# Recursive summarization
summaries = []
if len(docs) > 1:
    for doc in docs:
        summaries.append(summarizer.summarization(doc.page_content))
# %%
long_summary = ' '.join(summaries)

#%%
