
#%% Keyword extraction
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

class KeywordExtractor:
    def __init__(self, ngram_range=(1,1)):
        self.kw_extractor = KeyBERT()
        self.ngram_range = ngram_range

    def extract_keywords(self, text):
        # Use the extract_keywords function from the KeyBERT library
        keywords = self.kw_extractor.extract_keywords(text, keyphrase_ngram_range=self.ngram_range, stop_words='english')

        return keywords

#
# # Example usage
# keyword_extractor = KeywordExtractor()
#
# # Example 1: Extract keywords from text
# text = "Keyword extraction is a valuable task in natural language processing."
# keywords_text = keyword_extractor.extract_keywords(text)
# print("Keywords from text:", keywords_text)
