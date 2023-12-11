from newspaper import Article
from translation import Translator
#%%
class NewsArticle:
    def __init__(self, url):
        self.url = url
        self.article = Article(self.url)
        self.article.download()
        self.article.parse()
        # self.title = self.article.title
        # self.authors = self.article.authors
        # self.publish_date = str(self.article.publish_date)
        # self.text = self.article.text
        # self.image = self.article.top_image
        self.language_mapping = {
                                "ar": "Arabic",
                                "ru": "Russian",
                                "nl": "Dutch",
                                "de": "German",
                                "en": "English",
                                "es": "Spanish",
                                "fr": "French",
                                "he": "Hebrew",
                                "it": "Italian",
                                "ko": "Korean",
                                "no": "Norwegian",
                                "fa": "Persian",
                                "pl": "Polish",
                                "pt": "Portuguese",
                                "sv": "Swedish",
                                "hu": "Hungarian",
                                "fi": "Finnish",
                                "da": "Danish",
                                "zh": "Chinese",
                                "id": "Indonesian",
                                "vi": "Vietnamese",
                                "sw": "Swahili",
                                "tr": "Turkish",
                                "el": "Greek",
                                "uk": "Ukrainian",
                            }
        self.lang = self.language_mapping.get(self.article.meta_lang)
        # if self.lang != 'English':
        #     self.translate()

    # def translate(self):
    #     translator = Translator(source_lang=self.lang)
    #
    #     self.article.orig_title = self.article.title
    #     self.article.title = translator.translate(self.article.orig_title)
    #
    #     self.article.orig_authors = self.article.authors
    #     authors_translated = []
    #     for author in self.article.orig_authors:
    #         authors_translated.append(translator.translate(author))
    #     self.article.authors = authors_translated
    #
    #     self.article.orig_text = self.article.text
    #     self.article.text = translator.translate(self.article.orig_text)

#%%
## Example
#%%
# url = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html' # English News
# # url = 'https://www.bbc.com/arabic/live/67644287'
# news = NewsArticle(url)
#
# print('**NEWS ARTICLE**')
# print(f'Title : {news.article.title}')
# print(f'Author: {news.article.authors}')
# print(f'Text  : {news.article.text}')
