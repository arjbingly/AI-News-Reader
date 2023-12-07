from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartForConditionalGeneration
import pytorch
#%%
class Summarizer:
    def __init__(self):
        self.model = transformers.


# Load model directly


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')