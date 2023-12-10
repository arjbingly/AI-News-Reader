from bs4 import BeautifulSoup
import requests
from transformers import pipeline
from transformers import AutoTokenizer, BartForConditionalGeneration
from news_fetch import NewsArticle
import torch
class Summarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.max_length = 500
        self.min_length = 100
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    def summarization(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        input_ids = inputs.input_ids
        summary_ids = self.model.generate(input_ids, max_length=self.max_length, min_length=self.min_length, do_sample=False)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return summary

# # Example usage
# url = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'
# news = NewsArticle(url)
# article_text = news.article.text
#
# summarizer_instance = Summarizer()
# summary = summarizer_instance.summarization(article_text)
# print(summary)


