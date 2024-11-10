import os
import streamlit as st
import tempfile
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

# Initialize vectorstore to None initially
vectorstore = None

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
        text_content = [shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")]
        return " ".join(text_content)

# Caching function to load various file types
@st.cache_resource
def load_file(file_name, file_type):
    global vectorstore

    loaders = []
    if file_type == "pdf":
        loaders = [PyPDFLoader(file_name)]
    elif file_type == "docx":
        loader = DocxLoader(file_name)
        text = loader.load()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text.encode("utf-8"))
            temp_file_path = temp_file.name
        loaders = [TextLoader(temp_file_path)]
        
    elif file_type == "txt":
        loaders = [TextLoader(file_name)]
        
    elif file_type == "pptx":
        loader = PptxLoader(file_name)
        text = loader.load()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(text.encode("utf-8"))
            temp_file_path = temp_file.name
        loaders = [TextLoader(temp_file_path)]
        
    else:
        st.error("Unsupported file type.")
        return None

    # Create vectorstore index
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    ).from_loaders(loaders)
    
    vectorstore = index.vectorstore  # Set the global vectorstore
    return vectorstore

# Initialize API and Model
watsonx_api_key = os.getenv("WATSONX_API_KEY")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")

if not watsonx_api_key or not watsonx_project_id:
    st.error("API Key or Project ID is not set. Please set them as environment variables.")

prompt_template_br = PromptTemplate(
    input_variables=["context", "question"], 
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
I am a helpful assistant.

<|eot_id|>
{context}
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
"""
)

# Watsonx API setup
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": watsonx_api_key
}
params = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
    GenParams.MAX_NEW_TOKENS: 600,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
    GenParams.STOP_SEQUENCES: [],
    GenParams.REPETITION_PENALTY: 1
}
project_id = watsonx_project_id

# Sidebar for user input
with st.sidebar:
    st.title("Watsonx RAG with Multiple docs")
    model_name = st.selectbox("Model", ["meta-llama/llama-3-405b-instruct", "codellama/codellama-34b-instruct-hf", "ibm/granite-20b-multilingual"])
    uploaded_file = st.file_uploader("Upload file", accept_multiple_files=False, type=["pdf", "docx", "txt", "pptx"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.read())
        vectorstore = load_file(uploaded_file.name, file_type)

    def clear_messages():
        st.session_state.messages = []

    st.button('Clear messages', on_click=clear_messages)

# Model initialization
model = WatsonxLLM(model=Model(model_name, my_credentials, params, project_id, None, verify=False))

# Chat and Query System
if model:
    st.info(f"Model {model_name} ready.")
    chain = LLMChain(llm=model, prompt=prompt_template_br, verbose=True)

if chain and vectorstore:
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template_br},
        return_source_documents=False,
        verbose=True
    )
    st.info("Chat with document ready.")

# Store and display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask your question here", disabled=False if chain else True)

if prompt:
    st.chat_message("user").markdown(prompt)

    response_text = None
    if rag_chain:
        response_text = rag_chain.run(prompt).strip()
    else:
        response_text = chain.run(question=prompt, context="").strip("<|start_header_id|>assistant<|end_header_id|>").strip("<|eot_id|>")
    
    st.session_state.messages.append({'role': 'User', 'content': prompt })
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({'role': 'Assistant', 'content': response_text })
