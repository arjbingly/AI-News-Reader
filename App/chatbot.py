from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import LlamaCpp

import os
CUR_DIR = os.getcwd()
MODEL_DIR = CUR_DIR + os.sep + 'Models'
# %%
class ConversationalQA:
    def __init__(self):
        # self.model_path = "/home/ubuntu/Downloads/llama-2-13b-chat.Q5_K_M.gguf"
        self.model_path = MODEL_DIR+os.sep+"llama-2-13b-chat.Q5_K_M.gguf"
        self.embedding_model = 'sentence-transformers/all-MiniLM-L12-v2'
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,
                                                            chunk_overlap=50,
                                                            length_function=len,
                                                            is_separator_regex=False, )
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model,
                                                model_kwargs=model_kwargs,
                                                encode_kwargs=encode_kwargs,
                                                )
        self.k = 5
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True,
                                               output_key='answer')
        self.load_llm()

    def load_llm(self):
        n_gpu_layers = 40
        n_batch = 512
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            # callback_manager=callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

    def create_pipe(self):
        self.rag_pipeline = ConversationalRetrievalChain.from_llm(self.llm,
                                                                  self.retriever,
                                                                  chain_type='stuff',
                                                                  memory=self.memory,
                                                                  return_source_documents=True)

    def create_vector_db(self, text):
        pages = self.text_splitter.split_text(text)
        self.docs = self.text_splitter.create_documents(pages)
        self.db = Chroma.from_documents(self.docs, self.embeddings)
        self.retriever = self.db.as_retriever(search_kwargs={'k': self.k})

    def clear_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                               return_messages=True,
                                               output_key='answer')

    def infer(self, question, chat_history):
        # print(f'ChatHist: {self.memory.chat_memory}')
        # return self.rag_pipeline({'question': question, 'chat_history': chat_history})
        return self.rag_pipeline({'question': question})

# %%
# #E.G.
# from news_fetch import NewsArticle
#
# url = 'https://www.bbc.com/news/world-europe-63863088'  # long news article
# news = NewsArticle(url)
# text = news.article.text
# chatbot = ConversationalQA()
# chatbot.create_vector_db(text)
# chatbot.create_pipe()
# #%%
# question = 'Who were in pain?'
# results = chatbot.infer(question)
# #%%
# #%%
# dict_ = {i:doc.page_content for i,doc in enumerate(results['source_documents'])}
