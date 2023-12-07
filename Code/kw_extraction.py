
#%% Keyword exgtraction
# from transformers import BertTokenizer, BertForTokenClassification
# import torch
#
# def extract_keywords(text):
#     # Load pre-trained BERT tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForTokenClassification.from_pretrained('bert-base-uncased')
#
#     # Tokenize the input text
#     tokens = tokenizer(text, return_tensors='pt', truncation=True)
#
#     # Get the model's prediction
#     with torch.no_grad():
#         outputs = model(**tokens)
#
#     # Extract the predicted labels (token-level classification)
#     predicted_labels = torch.argmax(outputs.logits, dim=2)
#
#     # Map tokenized input back to words
#     words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0].tolist())
#
#     # Extract words with 'B-' (beginning of a keyword) label
#     keywords = [word for word, label_id in zip(words, predicted_labels[0].tolist()) if str(label_id).startswith('1')]
#
#     return keywords
#
# # Example usage
# text = "Keyword extraction is a valuable task in natural language processing."
# keywords = extract_keywords(text)
# print("Keywords:", keywords)

#%%
from keybert import KeyBERT
import requests
from bs4 import BeautifulSoup

class KeywordExtractor:
    def __init__(self):
        self.kw_extractor = KeyBERT()

    def extract_keywords(self, input_data, is_url=True):
        if is_url:
            text = self.fetch_text_from_url(input_data)
        else:
            text = input_data

        # Use the extract_keywords function from the KeyBERT library
        keywords = self.kw_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english')
        return keywords

    def fetch_text_from_url(self, url):
        # Fetch the text content from the URL using requests and BeautifulSoup
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the webpage (adjust based on the structure of the webpage)
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text

# Example usage
keyword_extractor = KeywordExtractor()

# Example 1: Extract keywords from text
text = "Keyword extraction is a valuable task in natural language processing."
keywords_text = keyword_extractor.extract_keywords(text, is_url=False)
print("Keywords from text:", keywords_text)

# Example 2: Extract keywords from URL
url = 'https://www.infoq.com/news/2023/12/aws-titan-image-generator/'
keywords_url = keyword_extractor.extract_keywords(url)
print("Keywords from URL:", keywords_url)
