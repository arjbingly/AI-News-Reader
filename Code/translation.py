from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# from news_fetch import NewsArticle

class Translator:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-one-mmt", target_lang = 'English', source_lang = None):
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.lang_names = {
    "Arabic": "ar_AR",     "Czech": "cs_CZ",     "German": "de_DE",     "English": "en_XX",
    "Spanish": "es_XX",     "Estonian": "et_EE",     "Finnish": "fi_FI",     "French": "fr_XX",
    "Gujarati": "gu_IN",     "Hindi": "hi_IN",     "Italian": "it_IT",     "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",     "Korean": "ko_KR",     "Lithuanian": "lt_LT",     "Latvian": "lv_LV",
    "Burmese": "my_MM",     "Nepali": "ne_NP",    "Dutch": "nl_XX",    "Romanian": "ro_RO",
    "Russian": "ru_RU",    "Sinhala": "si_LK",    "Turkish": "tr_TR",    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",    "Afrikaans": "af_ZA",    "Azerbaijani": "az_AZ",    "Bengali": "bn_IN",
    "Persian": "fa_IR",    "Hebrew": "he_IL",    "Croatian": "hr_HR",    "Indonesian": "id_ID",
    "Georgian": "ka_GE",    "Khmer": "km_KH",    "Macedonian": "mk_MK",    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",    "Marathi": "mr_IN",    "Polish": "pl_PL",    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",    "Swedish": "sv_SE",    "Swahili": "sw_KE",    "Tamil": "ta_IN",
    "Telugu": "te_IN",    "Thai": "th_TH",    "Tagalog": "tl_XX",    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",    "Xhosa": "xh_ZA",    "Galician": "gl_ES",    "Slovene": "sl_SI"
        }
        self.target_lang = self.get_lang_code(target_lang)
        self.source_lang = self.get_lang_code(source_lang)

        if target_lang is None:
            raise ValueError(f'Target Language not supported : {target_lang}')
        if source_lang is None:
            raise ValueError(f'Source Language not supported : {source_lang}')


    def get_lang_code(self, lang_key):
        # Use the dictionary to get the language code based on the provided key
        lang_code = self.lang_names.get(lang_key)
        return lang_code

    def translate(self, text):
        self.tokenizer.src_lang = self.source_lang
        self.tokenizer.tgt_lang = self.target_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded_text)
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, lang=self.target_lang)[0]
        return translated_text

# #%%
# # Example usage
# translator = NewsTranslator(source_lang="Korean")
# url = 'https://www.mk.co.kr/news/economy/10893700'
# news = NewsArticle(url)
# article_text = news.article.text
# translated_article = translator.translate(article_text)
# print(translated_article)
