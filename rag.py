from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
import pickle
from langchain_huggingface import HuggingFaceEmbeddings




def setup():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", max_retries=50)
    embeddings = HuggingFaceEmbeddings(model_name="avsolatorio/GIST-Embedding-v0")

    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=embeddings)
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)

    )
    docs = pickle.load(open("docs", 'rb'))
    retriever.add_documents(docs)

    
    system_prompt_1 = SystemMessage(content="""You are an helpful assistant for question-answering tasks. 
This is the conversation:

""")
    system_prompt_2 = ("system", """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{rag_context}
""")
    format_docs = RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs))
    
    rag_chain = retriever | format_docs
    
    chain =  ChatPromptTemplate.from_messages([system_prompt_1, MessagesPlaceholder(
        variable_name="chat_history"),system_prompt_2]) | llm | StrOutputParser()

    return chain, rag_chain

