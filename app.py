from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2  import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



def main():
    load_dotenv()
    # print("hello world!")
    # print("my API KEY :",os.env("OPENAI-KEY"))
    st.set_page_config(page_title="Ask your PDF")
    st.header('Sasta ChatGPT')

    pdf=st.file_uploader("upload your pdf ",type="pdf")

    #extract test

    if pdf is not None:
        pdf_reader=PdfReader(pdf)

        # intialize empty string 
        text=""
        #iterate over pdf files
        for page in pdf_reader.pages:
            text+=page.extract_text()

        # st.write(text)
        # split the document into small chunks

        text_splitter=CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index = True,
        )
        chunks=text_splitter.split_text(text)
        # st.write(chunks)

        # create word embeddings
        embeddings=OpenAIEmbeddings()
        database=FAISS.from_texts(chunks, embeddings)

        # now are we done with the embeddings 

        # its time to ask questions by the users related thier pdf context

        question=st.text_input("Ask questions about your pdf document")
        if question:
            related_docs=database.similarity_search(question)
            #st.write(answer)
            llm =OpenAI()
            chain=load_qa_chain(llm,chain_type='stuff')
            answer=chain.run(input_documents=related_docs,question=question)
            st.write(answer)

            

    
if __name__ == "__main__":
    main()
