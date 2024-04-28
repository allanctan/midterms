import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant

import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content=f"Please wait... initializing.", disable_feedback=True)
    await msg.send()

    loader = PyMuPDFLoader(
        "data/meta10k.pdf",
    )

    documents = loader.load()
    text=''
    for doc in documents:
        text+=doc.page_content

    # Split the text into chunks
    documents = text_splitter.split_documents(documents)

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create a Chroma vector store
    docsearch = await cl.make_async(Qdrant.from_documents)(
        documents, embeddings, location=":memory:"
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
    msg.content = f"Initialization successful. You can now ask questions!"
    await msg.update()

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
