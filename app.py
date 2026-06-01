import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import shutil

load_dotenv()

# ✅ FIX 1: Load API keys correctly from Streamlit secrets
google_api_key = st.secrets['GOOGLE_API_KEY']
groq_api_key = st.secrets['GROQ_API_KEY']

DB_PATH = "text_database"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_SUMMARY_WORDS = 8000

# ✅ FIX 2: Use ONE consistent embedding model everywhere
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"


def get_embedding_model():
    """Single function to get embedding model - ensures consistency"""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=google_api_key  # ✅ FIX 3: correct param name
    )


def summarizer(text, tokens):
    # ✅ FIX 4: Pass groq_api_key explicitly
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key)
    temp = PromptTemplate.from_template(
        '''You are a summarizer, you have to {tokens}\n\nand the given text is -- {text}--'''
    )
    parser = StrOutputParser()
    chain = temp | llm | parser
    return chain.invoke({'text': text, 'tokens': tokens})


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def store_text_to_vector_db(text):
    chunks = get_text_chunks(text)
    embedding_model = get_embedding_model()  # ✅ FIX 5: consistent model
    vectordb = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    vectordb.save_local(DB_PATH)


def get_context_from_vector_db(query):
    embedding_model = get_embedding_model()  # ✅ FIX 5: same model as storing
    vectordb = FAISS.load_local(
        DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    context_docs = vectordb.similarity_search(query)
    return [doc.page_content for doc in context_docs]


def generate_answer(query):
    context = get_context_from_vector_db(query)
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key)  # ✅ FIX 4

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful assistant. Use the context provided below to answer the question.
        If the answer cannot be found in the context, say "I don't know."

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
    except:
        return '''Sorry for the inconvenience. Here at InsightGenie, we are using a free API.
    The rate limit is exceeded, please try again after a few minutes ☺️'''


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
    elif file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""


from deep_translator import GoogleTranslator

languages = {
    'Afrikaans': 'af', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar',
    'Armenian': 'hy', 'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be',
    'Bengali': 'bn', 'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca',
    'Chinese (simplified)': 'zh-CN', 'Chinese (traditional)': 'zh-TW',
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl',
    'English': 'en', 'Estonian': 'et', 'Filipino': 'tl', 'Finnish': 'fi',
    'French': 'fr', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu',
    'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Indonesian': 'id',
    'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Korean': 'ko',
    'Malay': 'ms', 'Malayalam': 'ml', 'Marathi': 'mr', 'Nepali': 'ne',
    'Norwegian': 'no', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr',
    'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so',
    'Spanish': 'es', 'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta',
    'Telugu': 'te', 'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk',
    'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy',
    'Yoruba': 'yo', 'Zulu': 'zu'
}


def get_output(input_text, source, destination):
    translator = GoogleTranslator(
        source=languages[source], target=languages[destination]
    )
    return translator.translate(text=input_text)


def add_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #f1f1f1; color: #555;
        text-align: center; padding: 10px 0;
        font-size: 14px; z-index: 1000;
    }
    .footer a { color: #0366d6; text-decoration: none; margin: 0 10px; }
    .footer a:hover { text-decoration: underline; }
    </style>
    <div class="footer">
        Developed by A. K. Saini |
        <a href="https://github.com/aksaini2003" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/aashish-kumar-saini-03946b296/" target="_blank">LinkedIn</a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────
st.set_page_config(page_title="Insight Genie", layout="wide")
st.sidebar.title("Navigation Menu")

page = st.sidebar.radio(
    "Go to",
    ["Question Answering System", "Summarizer", "Language Translation"],
    index=0
)

add_footer()

# ──────────────────────────────────────────────
if page == "Question Answering System":
    st.title("📄 Question Answering Chatbot")

    # ✅ FIX 6: Initialize session state properly in one place
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False

    if st.button("🧹 Clear History"):
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        st.session_state.messages = []
        st.session_state.files_processed = False  # ✅ also reset this flag
        st.success("History and knowledge base cleared.")

    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )
    process_button = st.button("Process Files")

    if uploaded_files and process_button:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

        full_text = ""
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                extracted_text = extract_text_from_file(file)
                full_text += extracted_text + "\n"
            store_text_to_vector_db(full_text)

        st.success(f"{len(uploaded_files)} file(s) processed and indexed.")
        st.session_state.files_processed = True

    # ✅ FIX 7: Hard restriction — QnA only allowed after files are processed
    if st.session_state.files_processed and os.path.exists(DB_PATH):
        st.markdown("---")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        question = st.chat_input("Ask a question about the uploaded text...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.spinner("Generating answer..."):
                answer = generate_answer(question)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
    else:
        # ✅ Clear message shown when no document is uploaded yet
        st.info("⬆️ Please upload and process your document(s) first to start asking questions.")

# ──────────────────────────────────────────────
elif page == "Summarizer":
    st.title("📝 Text Summarizer")
    st.markdown(f"Paste or type up to **{MAX_SUMMARY_WORDS} words** below:")

    input_text = st.text_area("Enter your text here:", height=300)
    summary_size = st.selectbox(
        "Select summary size:",
        options=["Short (1-2 lines)", "Medium (1 paragraph)", "Detailed (multi-paragraph)"]
    )

    if st.button("Summarize"):
        word_count = len(input_text.split())
        if word_count > MAX_SUMMARY_WORDS:
            st.warning(f"Text exceeds {MAX_SUMMARY_WORDS} word limit. Currently: {word_count} words.")
        elif word_count == 0:
            st.info("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                size_prompt = {
                    "Short (1-2 lines)": "Write a very short 1-2 line summary.",
                    "Medium (1 paragraph)": "Write a concise summary in one paragraph.",
                    "Detailed (multi-paragraph)": "Write a detailed multi-paragraph summary covering all important points."
                }
                # ✅ FIX 8: Use consistent model name
                summary = summarizer(input_text, size_prompt[summary_size])
                st.subheader("Summary:")
                st.write(summary)

# ──────────────────────────────────────────────
elif page == "Language Translation":
    options = list(languages.keys())
    st.title('Free Language Translator')

    src_lang = st.selectbox('Select language of Input text', options, index=options.index('English'))
    dest_lang = st.selectbox('Select a language for the Translation', options, index=options.index('Hindi'))
    text_input = st.text_area('Text Area', placeholder='Enter your Text here')

    if st.button('Translate'):
        if text_input.strip():
            try:
                translated_text = get_output(text_input, source=src_lang, destination=dest_lang)
                st.text_area("Translated Text:", value=translated_text)
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")
        else:
            st.info("Please enter text to translate.")