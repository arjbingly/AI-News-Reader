# Code_Group5
# Arjun, HaeLee, Nayaeun

#%%
from bs4 import BeautifulSoup
import requests
from transformers import pipeline
# pip install transformers

# Function to fetch and summarize news article
def summarize_article(url):
    # Fetch the HTML content of the article
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the text content from the HTML
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.get_text() for p in paragraphs])

    # Use transformers library for summarization
    summarizer = pipeline("summarization")
    summary = summarizer(article_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=3)

    return summary[0]['summary']

# Example usage
news_url = "https://www.cnn.com/politics/live-news/donald-trump-jr-trial-testimony-11-13-23/index.html"
summary = summarize_article(news_url)
print("Summary:", summary)
# %%
