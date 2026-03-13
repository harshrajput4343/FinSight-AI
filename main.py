import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI — Research Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for Beautiful Dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    * { font-family: 'Inter', sans-serif !important; }
    
    /* Ensure all markdown text is visible (White/Light Grey) */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] li,
    .stMarkdown {
        color: #f1f5f9 !important;
    }

    .stApp {
        background: linear-gradient(145deg, #0a0a1a 0%, #0d1117 40%, #101828 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1724 0%, #131b2e 50%, #0f1724 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    /* ── Text Inputs ── */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.03em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Headers ── */
    h1 { color: #f1f5f9 !important; }
    h2, h3 { color: #e2e8f0 !important; }

    /* ── Status Text ── */
    .status-text {
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid #6366f1;
        padding: 12px 18px;
        border-radius: 0 10px 10px 0;
        color: #a5b4fc;
        font-size: 0.9rem;
        margin: 8px 0;
        animation: fadeInSlide 0.5s ease;
    }

    /* ── Answer Card ── */
    .answer-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .answer-card h3 {
        color: #a5b4fc !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 12px !important;
    }
    .answer-card p {
        color: #ffffff !important;
        font-size: 1.05rem !important;
        line-height: 1.75 !important;
    }

    /* ── Source Card ── */
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 12px;
        padding: 16px 22px;
        margin: 10px 0;
    }
    .source-card a {
        color: #818cf8 !important;
        text-decoration: none !important;
        font-size: 0.9rem;
        word-break: break-all;
    }
    .source-card a:hover {
        color: #a5b4fc !important;
        text-decoration: underline !important;
    }

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 60px 20px 40px;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.15rem;
        font-weight: 400;
        margin-top: 0;
        letter-spacing: 0.01em;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 50px;
        padding: 6px 18px;
        color: #a5b4fc;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-bottom: 20px;
    }

    /* ── Feature Cards ── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 30px 0;
    }
    .feature-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 12px;
    }
    .feature-title {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .feature-desc {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* ── Divider ── */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
        border: none;
        margin: 30px 0;
    }

    /* ── Animations ── */
    @keyframes fadeInSlide {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .processing-pulse {
        animation: pulse 1.5s infinite;
    }

    /* ── Sidebar Branding ── */
    .sidebar-brand {
        text-align: center;
        padding: 10px 0 25px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
        margin-bottom: 20px;
    }
    .sidebar-brand-icon {
        font-size: 2.2rem;
        margin-bottom: 6px;
    }
    .sidebar-brand-name {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-brand-tag {
        color: #475569;
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* ── URL Label Styling ── */
    .url-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .url-number {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
    }

    /* ── Question Input ── */
    .question-container {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 12px !important;
        color: #a5b4fc !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🔍</div>
        <div class="sidebar-brand-name">FinSight AI</div>
        <div class="sidebar-brand-tag">Powered by OpenRouter + Groq</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📰 Enter Article URLs")
    st.markdown(
        '<p style="color:#475569; font-size:0.82rem; margin-bottom:18px;">'
        "Paste up to 3 news articles to analyze</p>",
        unsafe_allow_html=True,
    )

    urls = []
    for i in range(3):
        st.markdown(
            f'<div class="url-label"><span class="url-number">{i+1}</span> Article URL</div>',
            unsafe_allow_html=True,
        )
        url = st.text_input(
            f"URL {i+1}",
            label_visibility="collapsed",
            placeholder=f"https://example.com/article-{i+1}",
            key=f"url_{i}",
        )
        urls.append(url)

    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    process_url_clicked = st.button("⚡ Process URLs", use_container_width=True)

    # Sidebar footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <p style="color:#334155; font-size:0.75rem; margin:0;">Built with</p>
        <p style="color:#475569; font-size:0.78rem; margin:4px 0 0;">
            🦜 LangChain · 🤖 Gemini · ⚡ FAISS
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ AI-POWERED RESEARCH</div>
    <h1 class="hero-title">FinSight AI</h1>
    <p class="hero-subtitle">Analyze news articles instantly with the power of Gemini AI.<br/>
    Ask questions, get answers with sources.</p>
</div>
""", unsafe_allow_html=True)

# Feature Cards
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">📥</div>
        <div class="feature-title">Load Articles</div>
        <div class="feature-desc">Paste any news URL and we'll extract the content automatically</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI Embeddings</div>
        <div class="feature-desc">Content is vectorized using Google's embedding model for semantic search</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💡</div>
        <div class="feature-title">Smart Answers</div>
        <div class="feature-desc">Ask anything and get precise answers with cited sources</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Folder path for FAISS index
# ─────────────────────────────────────────────
index_path = "faiss_index"

# ─────────────────────────────────────────────
# Processing Pipeline
# ─────────────────────────────────────────────
main_placeholder = st.empty()

if process_url_clicked:
    # Filter out empty URLs
    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.error("⚠️ Please enter at least one valid URL to process.")
    else:
        try:
            # Step 1: Load data
            print(f"\n[1/4] Loading URLs: {valid_urls}", flush=True)
            main_placeholder.markdown(
                '<div class="status-text processing-pulse">📥 <strong>Step 1/4</strong> — Loading article data...</div>',
                unsafe_allow_html=True,
            )
            loader = WebBaseLoader(valid_urls)
            data = loader.load()
            print(f"✅ Successfully loaded {len(data)} articles.", flush=True)

            # Step 2: Split text
            print("[2/4] Splitting text into chunks...", flush=True)
            main_placeholder.markdown(
                '<div class="status-text processing-pulse">✂️ <strong>Step 2/4</strong> — Splitting text into chunks...</div>',
                unsafe_allow_html=True,
            )
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","],
                chunk_size=1000,
                chunk_overlap=200,
            )
            docs = text_splitter.split_documents(data)
            print(f"✅ Split into {len(docs)} document chunks.", flush=True)

            # Step 3: Create embeddings (With Retry Logic)
            print("[3/4] Generating embeddings with Gemini...", flush=True)
            main_placeholder.markdown(
                '<div class="status-text processing-pulse">🧠 <strong>Step 3/4</strong> — Generating embeddings with Gemini...</div>',
                unsafe_allow_html=True,
            )
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            
            # Retry logic for embeddings
            vectorstore = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    break # Success!
                except Exception as e:
                    print(f"⚠️ Setup Error (Attempt {attempt+1}/{max_retries}): {str(e)}", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(2) # Wait 2 seconds before retrying
                    else:
                        raise e # Fail if all retries fail

            print("✅ Embeddings building complete.", flush=True)

            # Step 4: Save FAISS index
            print(f"[4/4] Saving vector index to: {index_path}", flush=True)
            main_placeholder.markdown(
                '<div class="status-text processing-pulse">💾 <strong>Step 4/4</strong> — Saving vector index...</div>',
                unsafe_allow_html=True,
            )
            vectorstore.save_local(index_path)
            print("✨ DONE: All processing completed successfully.\n", flush=True)

            main_placeholder.markdown(
                '<div class="status-text" style="border-left-color: #22c55e; color: #4ade80;">'
                "✅ <strong>All done!</strong> — Articles processed successfully. Ask a question below!</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            print(f"❌ ERROR DURING PROCESSING: {str(e)}", flush=True)
            error_msg = str(e)
            if "504" in error_msg:
                 st.error("❌ Google API Timeout (504). The articles might be too long. Try processing fewer URLs.", icon="⏳")
            else:
                 st.error(f"❌ Error during processing: {error_msg}")


# ─────────────────────────────────────────────
# Query Interface
# ─────────────────────────────────────────────
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom: 5px;">
    <span style="color:#6366f1; font-size:1.4rem;">💬</span>
    <span style="color:#e2e8f0; font-size:1.15rem; font-weight:600; margin-left:8px;">Ask a Question</span>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "Ask a question",
    label_visibility="collapsed",
    placeholder="🔎  What would you like to know from the articles?",
)

if query:
    if os.path.exists(index_path):
        try:
            # We need to initialize embeddings to load the vectorstore
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            # ─────────────────────────────────────────────
            # Multi-Provider Model Rotation
            # Each entry: (model_slug, api_key_env, api_base, display_name)
            # ─────────────────────────────────────────────
            models = [
                ("google/gemma-3n-e4b-it:free", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "Gemma 3n 4B (OpenRouter)"),
                ("llama-3.1-8b-instant",        "GROQ_API_KEY",       "https://api.groq.com/openai/v1", "Llama 3.1 8B (Groq)"),
                ("mistral-saba-24b",            "GROQ_API_KEY",       "https://api.groq.com/openai/v1", "Mistral Saba 24B (Groq)"),
            ]
            
            result = None
            last_error = None
            
            # Try each model in order until one succeeds
            with st.spinner("🤖 Thinking..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

                for model_slug, api_key_env, api_base, display_name in models:
                    try:
                        print(f"🔄 Attempting query with: {display_name} ({model_slug})...", flush=True)
                        current_llm = ChatOpenAI(
                            model=model_slug,
                            temperature=0.7,
                            max_tokens=500,
                            openai_api_key=os.getenv(api_key_env),
                            openai_api_base=api_base,
                            request_timeout=20,
                        )
                        chain = RetrievalQAWithSourcesChain.from_llm(
                            llm=current_llm,
                            retriever=retriever,
                        )
                        result = chain.invoke({"question": query})
                        st.session_state['current_model'] = display_name
                        print(f"✅ Success with: {display_name}", flush=True)
                        break  # Got a successful response, stop trying
                    except Exception as e:
                        print(f"⚠️ {display_name} failed: {str(e)}", flush=True)
                        last_error = e
                        continue  # Try next model

            if not result:
                # If all models failed
                raise last_error

            print(f"✅ Answer generated successfully with sources.", flush=True)

            # Display Answer
            raw_answer = result.get("answer", "No answer found.")
            
            # separating thought process from answer if present (DeepSeek style)
            thought_process = ""
            final_answer = raw_answer
            
            if "<think>" in raw_answer and "</think>" in raw_answer:
                parts = raw_answer.split("</think>")
                if len(parts) >= 2:
                    thought_process = parts[0].replace("<think>", "").strip()
                    final_answer = parts[1].strip()
            
            # Show thought process in expander if it exists
            if thought_process:
                with st.expander("💭 View Reasoning Process"):
                    st.markdown(thought_process)

            # Show which model was used
            active_model = st.session_state.get('current_model', 'AI Model')
            st.markdown(f'<div style="color: #64748b; font-size: 0.8rem; margin-bottom: 5px;">Generated by: {active_model}</div>', unsafe_allow_html=True)

            # Render the answer card with proper Markdown support and container
            st.markdown('<div class="answer-card"><h3>📌 Answer</h3>', unsafe_allow_html=True)
            st.markdown(final_answer)
            st.markdown('</div>', unsafe_allow_html=True)

            # Display Sources
            sources = result.get("sources", "")
            if sources:
                st.markdown(
                    '<div style="margin-top: 20px;"><h3 style="color:#a5b4fc; font-size:0.95rem; font-weight:600; '
                    'letter-spacing:0.08em; text-transform:uppercase;">'
                    "📎 Sources</h3></div>",
                    unsafe_allow_html=True,
                )
                sources_list = sources.strip().split("\n")
                for source in sources_list:
                    source = source.strip()
                    if source:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<a href="{source}" target="_blank">🔗 {source}</a>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            print(f"❌ ERROR DURING QUERY: {str(e)}", flush=True)
            st.error(f"❌ Error during query: {str(e)}")
    else:
        st.warning(
            "⚠️ No processed data found. Please enter URLs in the sidebar and click **Process URLs** first."
        )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; margin-top:60px; padding:20px 0;">
        <div class="glow-divider"></div>
        <p style="color:#334155; font-size:0.8rem; margin-top:20px;">
            FinSight AI — Powered by Google Gemini & LangChain
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
