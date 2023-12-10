from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
# %%
from news_fetch import NewsArticle
from transformers import pipeline

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
# %%
url = 'https://www.bbc.com/news/world-europe-63863088'  # long news article
news = NewsArticle(url)
text = news.article.text
print(len(text))

# %%
# loader = TextLoader(text)
# loader.load()
# %%
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
#                                                chunk_overlap=100,
#                                                length_function=len,
#                                                is_separator_regex = False,)
# text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)

model_name = 'bert-large-cased-whole-word-masking-finetuned-squad'
tokenizer = AutoTokenizer.from_pretrained(model_name)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(AutoTokenizer.from_pretrained('microsoft/mpnet-base'),
                                                                         chunk_size=512,
                                                                         chunk_overlap =100)


pages = text_splitter.split_text(text)
docs = text_splitter.create_documents(pages)

for i in range(4):
    print(f'{i} : {docs[i]}')
#%%
# embedding_model = 'sentence-transformers/all-MiniLM-L12-v2'
embedding_model = 'all-mpnet-base-v2' # uses Bert like Tokeniser
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   model_kwargs=model_kwargs,
                                   encode_kwargs=encode_kwargs,
                                   )


# #eg
# text = 'My name is Arjun'
# query_result = embeddings.embed_query(text)
#%%
db = Chroma.from_documents(docs, embeddings)

# #eg
# question = "Who is in pain?"
# searchDocs = db.similarity_search(question)
# for doc in searchDocs:
#     print(doc.page_content)
#%%


model_pipe = pipeline("question-answering",
                      model=model_name,
                      tokenizer=tokenizer,
                      return_tensors='pt')

llm = HuggingFacePipeline(pipeline=model_pipe,
                          model_kwargs={'temperature': 0.7,
                                        'max_length': 512})

#%%
retriever = db.as_retriever(search_kwargs={'k':3})


# qa = RetrievalQA.from_chain_type(llm=llm,
#                                  chain_type='refine',
#                                  retriever=retriever,
#                                  return_source_documents=False)
#%%
question = 'Who were in pain?'
context_result = retriever.get_relevant_documents(question)

# result = qa.run({'query': question})
# print(result['result'])
#%%
question_input = []
context_input = []
for context in context_result:
    question_input.append(question)
    context_input.append(context.page_content)


results = model_pipe(question=question_input, context=context_input)
print(results)
