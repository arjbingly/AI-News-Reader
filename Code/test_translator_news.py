from news_fetch import NewsArticle
from translation import Translator

# Example usage
url = 'https://www.mk.co.kr/news/economy/10893700'
news = NewsArticle(url)
translator = Translator(source_lang=news.lang)
article_text = news.article.text
translated_article = translator.translate(article_text)
translated_title = translator.translate(news.article.title)
translated_authors = translator.translate(' ,'.join(news.article.authors))

print(translated_title)
print(translated_authors)
print(translated_article)

