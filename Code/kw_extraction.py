
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

def fetch_text_from_url(url):
    # Fetch the text content from the URL using requests and BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text from the webpage (adjust based on the structure of the webpage)
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

# Example usage
url = 'https://www.infoq.com/news/2023/12/aws-titan-image-generator/'
text_from_url = fetch_text_from_url(url)

kw_extractor = KeyBERT()

# Use the extract_keywords function from the previous example
keywords = kw_extractor.extract_keywords(text_from_url, keyphrase_ngram_range=(1, 1), stop_words='english')
print("Keywords:", keywords)
