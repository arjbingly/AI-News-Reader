from transformers import AutoTokenizer, BartForConditionalGeneration
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Summarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.max_length = 300
        self.min_length = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = 'cpu'
        self.model = self.model.to(self.device)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer,
                                                                                       chunk_size=512,
                                                                                       chunk_overlap=10)

    def split_text(self, text):
        pages = self.text_splitter.split_text(text)
        self.docs = self.text_splitter.create_documents(pages)
    def summarize(self, text):
        self.split_text(text)
        summaries = []
        for doc in self.docs:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            input_ids = inputs.input_ids
            summary_ids = self.model.generate(input_ids,
                                              max_length=self.max_length,
                                              min_length=self.min_length,
                                              do_sample=False)
            output = self.tokenizer.decode(summary_ids[0],
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
            summaries.append(output)
        summary = ' '.join(summaries)

        return summary

#%%
# Example usage
# from news_fetch import NewsArticle
# url = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'
# news = NewsArticle(url)
# article_text = news.article.text
#
# summarizer_instance = Summarizer()
# summary = summarizer_instance.summarize(article_text)
# print(summary)
