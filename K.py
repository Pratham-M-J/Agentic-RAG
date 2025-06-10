import os
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import Settings
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
import tempfile


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


st.set_page_config(page_title="Gemini PDF Q&A", page_icon="📄")
st.title("📚 Ask Your PDF")
st.caption("Built with 💖 by Pratham.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

query_input = st.text_input("Enter Query:", placeholder="E.g., Summarize the document")

if uploaded_file and query_input:
    with st.spinner("Processing..."):

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        llm = Gemini(
            model="models/gemini-2.0-flash",
            api_key=GOOGLE_API_KEY
        )

        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        Settings.llm = llm
        Settings.embed_model = embed_model

        
        documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()

        splitter = SentenceSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separator='\n'
        )

        chunks = splitter.get_nodes_from_documents(documents)

       
        summary_index = SummaryIndex(chunks)
        vector_index = VectorStoreIndex(chunks)

        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            name="summary_tool",
            description="Use this for summarizing documents or getting an overview"
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name="vector_tool",
            description="Use this for specific fact retrieval, semantic search or keyword-based questions"
        )

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=True
        )

     
        response = query_engine.query(query_input)

        
        st.subheader("📥 Response")
        st.markdown(str(response))

        
        os.remove(tmp_path)

elif not uploaded_file:
    st.info("Please upload a PDF to get started.")
elif not query_input:
    st.info("Enter a question to ask your PDF.")

st.markdown("---")
st.markdown("🧠 Tip: Try asking summary-based or fact-based questions to see different routing behaviors!")
