import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

from dotenv import load_dotenv
import shutil
# Load environment variables
load_dotenv()

#in the streamlit application we have to get the api key
google_api_key = st.secrets['GOOGLE_API_KEY']  # Use Streamlit secrets for securit
groq_api_key=st.secrets['GROQ_API_KEY']  # Use Streamlit secrets for security
# Constants
DB_PATH = "text_database"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_SUMMARY_WORDS=8000

def summarizer(text,tokens):
    #here we are going to use the groq for the summarizeation task
    llm=ChatGroq(model='Llama-3.3-70b-versatile')

    
    
    temp=PromptTemplate.from_template('''You are a summarizer, you have to  {tokens}\n\n
                                      
                                      and the give text is -- {text}--''')
    
    parser=StrOutputParser()
    chain=temp|llm|parser
    return chain.invoke({'text':text,'tokens':tokens})

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def store_text_to_vector_db(text):
    chunks = get_text_chunks(text)
    
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',api_key=google_api_key)
    # embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    vectordb.save_local(DB_PATH)

def get_context_from_vector_db(query):
    #goolge embeddings is faster so here we are using the google_embedding  model
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004',api_key=google_api_key)
    # embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    context_docs = vectordb.similarity_search(query)
    return [doc.page_content for doc in context_docs]

def generate_answer(query):
    context = get_context_from_vector_db(query)
    llm = ChatGroq(model='Llama-3.3-70b-versatile')

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Use the context provided below to answer the question. 
        If the answer cannot be found in the context, say \"I don't know.\"

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    chain = prompt_template | llm | StrOutputParser()
    try:
        return chain.invoke({"context": "\n".join(context), "question": query})
        # raise ValueError('no response is generated ')
    except:
        return '''Sorry for the inconvenience. Here at InsightGenie, we are using a free API. 
    And the rate limit is exceeded, please try again after a few minutes☺️
    '''

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/msword"]:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    return ""  # Fallback 
from deep_translator import GoogleTranslator
# from googletrans import Translator
# translator=Translator()

languages={'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cebuano': 'ceb', 'Chichewa': 'ny', 'Chinese (simplified)': 'zh-CN', 'Chinese (traditional)': 'zh-TW', 'Corsican': 'co', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi', 'French': 'fr', 'Frisian': 'fy', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian creole': 'ht', 'Hausa': 'ha', 'Hawaiian': 'haw', 'Hebrew': 'he', 'Hindi': 'hi', 'Hmong': 'hmn', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish (kurmanji)': 'ku', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Myanmar (burmese)': 'my', 'Nepali': 'ne', 'Norwegian': 'no', 'Odia': 'or', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Samoan': 'sm', 'Scots gaelic': 'gd', 'Serbian': 'sr', 'Sesotho': 'st', 'Shona': 'sn', 'Sindhi': 'sd', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tajik': 'tg', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'}


def get_output(input_text,source,destination):
    translator = GoogleTranslator(source=languages[source], target=languages[destination])
    output = translator.translate(text=input_text)
    return output

#lets add the language translation ability to our text genie application
def translation():
    options = list(languages.keys())

    st.title('Free Language Translator')
    src_lang = st.selectbox('Select language of Input text', options, index=options.index('English'))
    dest_lang = st.selectbox('Select a language for the Translation', options, index=options.index('Hindi'))

    text_input = st.text_area('Text Area', placeholder='Enter your Text here')

    if st.button('Translate'):
        if text_input.strip() != '':
            try:
                translated_text = get_output(text_input, source=src_lang, destination=dest_lang)
                st.text_area("Translated Text:", value=translated_text)
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
        else:
            st.info("Please enter text to translate.")

    #lets add the both  source and the destination language options in our application
    #lets test our function 
def add_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #555;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        z-index: 1000;
    }
    .footer a {
        color: #0366d6;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Developed by A. K. Saini | 
        <a href="https://github.com/aksaini2003" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/aashish-kumar-saini-03946b296/" target="_blank">LinkedIn</a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
# Streamlit UI

st.set_page_config(page_title="Insight Genie", layout="wide")

st.sidebar.title("Navigation Menu")
st.markdown("""
    <style>
    /* Style the sidebar title */
    .sidebar .sidebar-content {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Style for sidebar header and radio buttons */
    section[data-testid="stSidebar"] .css-1v0mbdj,  /* sidebar title */
    section[data-testid="stSidebar"] .css-1cpxqw2,  /* radio container */
    section[data-testid="stSidebar"] label {
        font-size: 20px !important;
        font-weight: 600;
        color: #2C3E50;
        padding: 8px 4px;
        margin-bottom: 6px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    /* Hover effect */
    section[data-testid="stSidebar"] label:hover {
        background-color: #f0f0f5;
        color: #1A5276;
        cursor: pointer;
    }

    /* Highlight selected option */
    section[data-testid="stSidebar"] input:checked + div {
        background-color: #1A5276 !important;
        color: white !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Go to", ["Question Answering System", "Summarizer",'Language Translation'], index=0)
#for enhancing the navigation menu


add_footer()
if page == "Question Answering System":
    st.title("📄 Question Answering Chatbot")
    if st.button("🧹 Clear History"):
    # Delete vector DB folder
        if os.path.exists("text_database"):
            shutil.rmtree("text_database")

    # Clear chat messages and file processing status from session state
        st.session_state.messages = []
        st.session_state.files_processed = False

        st.success("History and knowledge base cleared.")



    
    # if st.button("🧹 Clear History"):
    # # Delete vector DB folder
    #     if os.path.exists("text_database"):
    #         shutil.rmtree("text_database")
    # # Clear chat messages from session state
    #         st.session_state.messages = []
    #         st.success("History and knowledge base cleared.")
    uploaded_files = st.file_uploader("Upload one or more files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    process_button = st.button("Process Files")

    full_text = ""

    # uploaded_files = st.file_uploader("Upload one or more files", type=["txt","pdf","docx"], accept_multiple_files=True)

    # process_button = st.button("Process Files")

    # full_text = ""

    # if uploaded_files and process_button:
    #     for file in uploaded_files:
    #         full_text += file.read().decode("utf-8") + "\n"
    #     store_text_to_vector_db(full_text)
    #     st.success(f"{len(uploaded_files)} file(s) processed and indexed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if uploaded_files and process_button:
        if os.path.exists("text_database"):
            shutil.rmtree("text_database")

        for file in uploaded_files:
            extracted_text = extract_text_from_file(file)
            full_text += extracted_text + "\n"
        
        # Store to vector DB (your function)
        store_text_to_vector_db(full_text)

        st.success(f"{len(uploaded_files)} file(s) processed and indexed.")
            
        
        
        
        # for file in uploaded_files:
        #     extracted_text = extract_text_from_file(file)
        #     full_text += extracted_text + "\n"
        
        # # Store to vector DB (your function)
        # store_text_to_vector_db(full_text)
        
        # st.success(f"{len(uploaded_files)} file(s) processed and indexed.")
        st.session_state.files_processed = True

    # Ensure chatbot is available after file upload
    if os.path.exists("text_database") or st.session_state.get("files_processed", False):
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        question = st.chat_input("Ask a question about the uploaded text...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            answer = generate_answer(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    else:
        st.info("Upload and process files to start asking questions.")

elif page == "Summarizer":
    st.title("📝 Text Summarizer")

    st.markdown(f"Paste or type up to **{MAX_SUMMARY_WORDS} words** below:")

    input_text = st.text_area("Enter your text here:", height=300)

    summary_size = st.selectbox(
        "Select summary size:",
        options=["Short (1-2 lines)", "Medium (1 paragraph)", "Detailed (multi-paragraph)"]
    )

    summarize_button = st.button("Summarize")

    if summarize_button:
        word_count = len(input_text.split())
        if word_count > MAX_SUMMARY_WORDS:
            st.warning(f"Text exceeds {MAX_SUMMARY_WORDS} word limit. Currently: {word_count} words.")
        elif word_count == 0:
            st.info("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):

                # Add instructions based on summary size
                size_prompt = {
                    "Short (1-2 lines)": "Write a very short 1-2 line summary.",
                    "Medium (1 paragraph)": "Write a concise summary in one paragraph.",
                    "Detailed (multi-paragraph)": "Write a detailed multi-paragraph summary covering all important points."
                }

                llm = ChatGroq(model="llama3-70b-8192")
                summary=summarizer(input_text,size_prompt[summary_size])

                st.subheader("Summary:")
                st.write(summary)
elif page=='Language Translation':
        options = list(languages.keys())

        st.title('Free Language Translator')
        src_lang = st.selectbox('Select language of Input text', options, index=options.index('English'))
        dest_lang = st.selectbox('Select a language for the Translation', options, index=options.index('Hindi'))

        text_input = st.text_area('Text Area', placeholder='Enter your Text here')

        if st.button('Translate'):
            if text_input.strip() != '':
                try:
                    translated_text = get_output(text_input, source=src_lang, destination=dest_lang)
                    st.text_area("Translated Text:", value=translated_text)
                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")
            else:
                st.info("Please enter text to translate.")
