import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant
from langchain.chains import (
    ConversationalRetrievalChain,
)
# from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub

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

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Let's build the chain
    primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    document_chain = create_stuff_documents_chain(primary_qa_llm, retrieval_qa_prompt)

    qdrant_vector_store = Qdrant.from_documents(
        documents,
        embeddings,
        location=":memory:",
        collection_name="Meta10k",
    )
    retriever = qdrant_vector_store.as_retriever()
    new_advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=primary_qa_llm)
    new_retrieval_chain = create_retrieval_chain(new_advanced_retriever, document_chain)

    # chain = ConversationalRetrievalChain.from_(
    #     primary_qa_llm,
    #     retriever=new_advanced_retriever,
    #     memory=memory,
    #     return_source_documents=True,
    # )

    # Let the user know that the system is ready
    msg.content = f"Okay, I'm ready. You may ask anything about Meta 10K (2023) filings..."
    await msg.update()

    cl.user_session.set("chain", new_retrieval_chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    response = chain.invoke({"input" : message.content})
    answer = response["answer"]
    print(response)

    # source_documents = response["source_documents"]  # type: List[Document]

    # text_elements = []  # type: List[cl.Text]

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(page)}"
    #     else:
    #         answer += "\nNo sources found"

    await cl.Message(content=answer).send()
