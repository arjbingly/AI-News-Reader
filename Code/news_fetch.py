from newspaper import Article
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

#%%
## Example
#%%
url = 'https://www.cnn.com/2023/10/03/europe/nobel-prize-physics-electrons-flashes-light-intl-scn/index.html'
news = NewsArticle(url)
#%%
print('**NEWS ARTICLE**')
print(f'Title : {news.article.title}')
print(f'Author: {news.article.authors}')
print(f'Text  : {news.article.text}')
