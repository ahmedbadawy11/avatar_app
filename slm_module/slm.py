import os
import threading
import subprocess
from typing import List, Tuple

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.embeddings.base import Embeddings


def start_ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])


class SLM:
    def __init__(self, model_name: str = "gemma2:2b-instruct-fp16", embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", docs_path: str = "words/en", collection_name: str = "en", persist_directory: str = "vectorstore", window_size: int = 4):
        self.window_size = window_size
        self.store = {}
        
        # Start Ollama in a separate thread
        threading.Thread(target=start_ollama, daemon=True).start()
        
        self.llm = ChatOllama(model=model_name, temperature=0.5)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        self.client = chromadb.Client()
        self.retriever = self.load_documents(doc_folder_path=docs_path, persist_directory=persist_directory, collection_name=collection_name)
        self.rag_chain = None
        self._initialize_chain()

    def _initialize_chain(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question in United Arab Emirates spoken language."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a chatbot assistant that answers questions using the relevant information provided in the context.
                Communicate in United Arab Emirates spoken language only.
                If you can't find the answer in the context, ask the user for more clarification. Do not improvise.

                Context: {context}
            """),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.rag_chain = RunnableWithMessageHistory(
            create_retrieval_chain(
                create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt),
                create_stuff_documents_chain(self.llm, qa_prompt),
            ),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def load_documents(self, doc_folder_path: str, persist_directory: str, collection_name: str):
        documents = []
        for file in os.listdir(doc_folder_path):
            if file.endswith('.doc') or file.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(os.path.join(doc_folder_path, file))
                documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunked_documents = text_splitter.split_documents(documents)

        if collection_name not in self.client.list_collections():
            self.client.create_collection(collection_name)
        
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding_model,
            persist_directory=persist_directory
        )
        
        return vectordb.as_retriever(search_kwargs={"k": 3})

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def trim_messages(self, chain_input: dict) -> bool:
        chat_history = self.get_session_history(chain_input["session_id"])
        if len(chat_history.messages) > self.window_size:
            chat_history.messages = chat_history.messages[-self.window_size:]
            return True
        return False

    def ask(self, question: str, session_id: str) -> str:
        response = (RunnablePassthrough.assign(messages_trimmed=self.trim_messages) | self.rag_chain).invoke(
            {"input": question, "session_id": session_id},
            {"configurable": {"session_id": session_id}}
        )
        return response['answer']