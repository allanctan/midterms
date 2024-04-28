import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # let's load the file
    loader = PyMuPDFLoader(
        "data/meta10k.pdf",
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50
    )

    texts = text_splitter.split_documents(documents)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Qdrant vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = await cl.make_async(Qdrant.from_texts)(
      texts, embeddings, location=":memory:", collection_name="Meta10k", metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg = cl.Message(content=f"Please ask me anything about Meta 10K (2023) filings...")
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
