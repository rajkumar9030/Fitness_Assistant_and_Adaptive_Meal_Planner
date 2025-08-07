import streamlit as st
import os
import faiss
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# === CONFIGURATION ===
GOOGLE_API_KEY = "AIzaSyD-TE2m5Ez2J6xJZEOgrZOxGk2CR5zJkak"  
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# === FUNCTION TO LOAD INDEX ===
@st.cache_resource
def load_index(file_path):
    loader = TextLoader(file_path, encoding="utf8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore.as_retriever()

# === BUILD RAG CHAIN ===
def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template("""
    You are a certified expert in nutrition and adaptive meal planning. Your task is to analyze the user's uploaded medical reports and lifestyle documents to generate a phase-wise, personalized meal plan. Begin by extracting key health indicators (such as blood sugar, cholesterol, thyroid levels, BMI, and nutritional deficiencies), identifying any medical conditions, allergies, or dietary restrictions. Based on this analysis, design a structured meal plan divided into phases—such as Stabilization, Nutritional Recovery, and Maintenance—each tailored to the user’s health goals, preferences, and activity levels. For each phase, provide balanced daily meal schedules including breakfast, lunch, dinner, and snacks, ensuring nutritional adequacy, cultural suitability, and practicality. Your recommendations must be adaptive—continually learning from previous inputs and feedback to refine and optimize future suggestions in alignment with the user’s evolving health needs.

    Question: {question}
    Context: {context}
    Answer:
    """)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# === PAGE LAYOUT ===
st.set_page_config(page_title="Adaptive Meal Planner", layout="centered")
st.markdown("<h1 style='text-align: center;'>🥗 AI-Powered Adaptive Meal Planner</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px;'>Upload your health report and let our AI generate a <b>personalized, phase-wise meal plan</b> just for you!</p>", unsafe_allow_html=True)
st.markdown("---")

# === FILE UPLOAD ===
st.markdown("###  Upload Your Health Report (.txt only)")
uploaded_file = st.file_uploader("", type="txt")

# === FILE PROCESSING ===
if uploaded_file:
    with open("data/uploaded_doc.txt", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner(" Indexing document... Please wait."):
        retriever = load_index("data/uploaded_doc.txt")
        rag_chain = build_rag_chain(retriever)

    st.success("Document indexed successfully!")

    # === USER QUERY INPUT ===
    st.markdown("###  Ask a question (or type: *Prepare a personalised meal planner*)")
    query = st.text_input("Type your query below:")

    if st.button(" Get Answer") and query:
        with st.spinner(" Thinking..."):
            answer = rag_chain.invoke(query)

        st.markdown("### Answer")
        st.success(answer)
else:
    st.info(" Please upload a `.txt` file to begin.")
