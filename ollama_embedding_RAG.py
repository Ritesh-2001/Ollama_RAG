#pip install -U langchain langchain-community langchain-core BeautifulSoup4 tiktoken chromadb
'''
First download ollama form ollama.com/download and then start it by running the ollama.setup.exe file
Before executing notebook, do following in cmd
enter virtual environment then type python ollama_embedding_RAG.py in cmd
ollama pull llama2
ollama pull mistral
ollama pull nomic-embed-text
'''

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

# load model mistral or llama2
model_local = ChatOllama(model="llama2") # can also add attributes like temperature (0 to 1, more temperature means more creative)
## Before RAG

# 3. Before RAG
print("\n########\nBefore RAG\n")
before_rag_template = "What is {topic}?"
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()
print(before_rag_chain.invoke({"topic": "Ollama"}))

## AFTER RAG 

# 1. Split data into chunks
urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

# 4. After RAG output
print("\n########\nAfter RAG\n")
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

# Here retriever is the context based on which model will answer the question
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("What is Ollama?"))

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()


'''
* Relevant documents are gathered from the web using URLs.
* Text from these documents is split into smaller chunks for efficient processing.
* Each chunk is converted into an embedding using the "nomic-embed-text" model.
* Embeddings are stored in a "Chroma" vector store for efficient retrieval.

NOTE: nomic-embed-text is a 8192 context length text encoder that surpasses OpenAI text-embedding-ada-002 and text-embedding-3-small performance on short and long context tasks.
'''
