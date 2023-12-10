from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
#%%
from news_fetch import NewsArticle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


#%%
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#%%
# GPU
n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/ubuntu/Downloads/llama-2-13b-chat.Q5_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    # callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# %%
url = 'https://www.bbc.com/news/world-europe-63863088'  # long news article
news = NewsArticle(url)
text = news.article.text
print(len(text))

#%%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,
                                               chunk_overlap=50,
                                               length_function=len,
                                               is_separator_regex = False,)

#%%
pages = text_splitter.split_text(text)
docs = text_splitter.create_documents(pages)
#%%
#%%
embedding_model = 'sentence-transformers/all-MiniLM-L12-v2'
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                   model_kwargs=model_kwargs,
                                   encode_kwargs=encode_kwargs,
                                   )
#%%
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever(search_kwargs={'k':5})
#%%
# RAG
# rag_pipeline = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type='stuff',
#     retriever=retriever
# )
# #%%
# question = 'What is the issue?'
# results = rag_pipeline(question)
#%%
# Conversational RAG
memory = []
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
rag_pipeline = ConversationalRetrievalChain.from_llm(llm,
                                                     retriever,
                                                     chain_type = 'stuff',
                                                     memory = memory,
                                                     return_source_documents=True)
#%%
question = 'What is the issue?'
result = rag_pipeline({"question": question})
#%%
def infer(question):
    return rag_pipeline({'question': question})