import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from pptx import Presentation
from docx import Document

# Load environment variables
load_dotenv()
watsonx_api_key = os.getenv("API_KEY")
watsonx_project_id = os.getenv("PROJECT_ID")
watsonx_url = "https://us-south.ml.cloud.ibm.com"

if not watsonx_api_key or not watsonx_project_id:
    st.error("API Key or Project ID is not set. Please set them as environment variables.")

# Custom loader for DOCX files
class DocxLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        document = Document(self.file_path)
        text_content = [para.text for para in document.paragraphs]
        return " ".join(text_content)

# Custom loader for PPTX files
class PptxLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        presentation = Presentation(self.file_path)
        text_content = []
        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_content.append(shape.text)
        return " ".join(text_content)

# Caching function to load various file types
@st.cache_resource
def load_file(file_name, file_type):
    loaders = []
    if file_type == "pdf":
        loaders = [PyPDFLoader(file_name)]
    elif file_type == "docx":
        text = DocxLoader(file_name).load()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text.encode("utf-8"))
            loaders = [TextLoader(temp_file.name)]
    elif file_type == "txt":
        loaders = [TextLoader(file_name)]
    elif file_type == "pptx":
        text = PptxLoader(file_name).load()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text.encode("utf-8"))
            loaders = [TextLoader(temp_file.name)]
    else:
        st.error("Unsupported file type.")
        return None

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    ).from_loaders(loaders)
    return index

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
I am a helpful assistant.
<|eot_id|>
{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>"""
)

# Sidebar settings
with st.sidebar:
    st.title("Watsonx RAG Demo")
    st.info("Setting up Watsonx configuration")
    model_name = st.selectbox("Model", ["meta-llama/llama-3-405b-instruct", "codellama/codellama-34b-instruct-hf", "ibm/granite-20b-multilingual"])
    max_new_tokens = st.slider("Max output tokens", min_value=100, max_value=1000, value=300, step=100)
    decoding_method = st.radio("Decoding Method", [DecodingMethods.GREEDY.value, DecodingMethods.SAMPLE.value])
    uploaded_file = st.file_uploader("Upload file (PDF, DOCX, TXT, PPTX)", accept_multiple_files=False)

    if uploaded_file:
        with st.spinner("Processing file..."):
            file_type = uploaded_file.name.split('.')[-1].lower()
            index = load_file(uploaded_file.name, file_type)
        st.success("File processed and indexed")

# Watsonx Model setup
params = {
    GenParams.DECODING_METHOD: decoding_method,
    GenParams.MAX_NEW_TOKENS: max_new_tokens,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

st.info("Initializing Watsonx model...")
model = None
try:
    model = WatsonxLLM(
        model=Model(model_name, {"url": watsonx_url, "apikey": watsonx_api_key}, params, project_id=watsonx_project_id)
    )
    st.success(f"Model [{model_name}] is ready")
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")

# Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User Input
prompt = st.chat_input("Ask your question here", disabled=not model)

# Handle User Query
if prompt:
    st.chat_message("user").markdown(prompt)

    # Generate Response
    response_text = ""
    if index:
        rag_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=index.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            verbose=True
        )
        response_text = rag_chain.run(prompt)
    else:
        llm_chain = LLMChain(llm=model, prompt=prompt_template)
        response_text = llm_chain.run(context="", question=prompt)

    response_text = response_text.strip()
    st.session_state.messages.append({'role': 'User', 'content': prompt})
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({'role': 'Assistant', 'content': response_text})
