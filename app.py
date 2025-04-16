import streamlit as st
import pickle
import os
import PyPDF2 as pdf
import docx
import re
import google.generativeai as genai
from dotenv import load_dotenv

# ========== Setup ==========
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

svc_model = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\clf.pkl', 'rb'))
tfidf = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\tfidf.pkl', 'rb'))
le = pickle.load(open(r'C:\Users\harsh\PycharmProjects\PythonProject\encoder.pkl', 'rb'))

# ========== Utils ==========
def clean_resume(txt):
    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"RT|cc", " ", txt)
    txt = re.sub(r"#\S+|@\S+", " ", txt)
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"[^\x00-\x7f]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            reader = pdf.PdfReader(file)
            return ''.join([page.extract_text() or "" for page in reader.pages])
        elif ext == 'docx':
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == 'txt':
            try:
                return file.read().decode('utf-8')
            except:
                file.seek(0)
                return file.read().decode('latin-1')
    except:
        return None

def predict_category(text):
    cleaned = clean_resume(text)
    vectorized = tfidf.transform([cleaned]).toarray()
    pred = svc_model.predict(vectorized)
    return le.inverse_transform(pred)[0]

def get_gemini_response(resume_text, job_description):
    prompt = f"""
    Act like a skilled ATS in tech hiring. Evaluate this resume based on the given job description.start with how much the profile matches the job description and then a small summary of max 2 to 3 line and then what are missing keywords in that profile or the skills only points not described and nothing else.
    {{
        "Profile Summary": ""
    }}

    Resume: {resume_text}
    JD: {job_description}
    """
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text

def render_feedback(parsed):
    try:
        match_percent = parsed["JD Match"]
        if "%" in match_percent:
            match_val = int(match_percent.replace('%', '').strip())
        else:
            match_val = int(match_percent.strip())
    except:
        match_val = 0

    missing_keywords = ', '.join(parsed['MissingKeywords']) or 'None 🚀'
    st.markdown(f"""
        <div style="
            background: #ffffff;
            border: 1px solid #ddd;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        ">
            <h4 style="color:#4a90e2;">📊 JD Match</h4>
            <div style="background-color:#e6f0ff; border-radius:10px; overflow:hidden;">
                <div style="width:{match_val}%; background:#4a90e2; padding:6px 12px; color:white; font-weight:bold;">
                    {match_percent}
                </div>
            </div>
            <h4 style="color:#e67e22; margin-top: 1rem;">🔑 Missing Keywords</h4>
            <p>{missing_keywords}</p>
            <h4 style="color:#2c3e50;">📝 Profile Summary</h4>
            <p style="font-style: italic;">{parsed['Profile Summary']}</p>
        </div>
    """, unsafe_allow_html=True)

# ========== Page Config & Style ==========
st.set_page_config("Smart Resume Parser", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f9fbfd;
        }
        h1, h2, h3, h4 {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(to right, #4a90e2, #0072ff);
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Header ==========
st.markdown("""
    <div style="text-align: center; padding: 20px; border-radius: 10px;
                background: linear-gradient(to right, #1e3c72, #2a5298); color: white;">
        <h1 style="margin-bottom: 0;">🤖 Smart Resume Parser</h1>
        <p style="font-size: 16px;">Upload multiple resumes, match them with your target role, and get Gemini-powered feedback!</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========== Sidebar Inputs ==========
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🎯 Select Target Role")
    job_roles = le.classes_
    display_roles = [r.replace("_", " ").title() for r in job_roles]
    selected_index = st.selectbox("Job Role", range(len(job_roles)), format_func=lambda i: display_roles[i])
    target_label = job_roles[selected_index]

with col2:
    st.subheader("📋 Paste Job Description")
    job_description = st.text_area("Job Description", height=200)

st.markdown("---")

# ========== Upload ==========
st.subheader("📄 Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# ========== Analyze Button ==========
if st.button("🚀 Analyze Resumes"):
    if not uploaded_files or not job_description.strip():
        st.warning("Please upload resumes and provide job description.")
    else:
        matched, all_results = [], []
        with st.spinner("Processing..."):
            for file in uploaded_files:
                try:
                    text = extract_text(file)
                    if not text:
                        raise ValueError("Unable to extract text.")
                    category = predict_category(text)

                    feedback = None
                    if category == target_label:
                        feedback = get_gemini_response(text, job_description)

                    result = (file.name, category, feedback)
                    all_results.append(result)
                    if category == target_label:
                        matched.append(result)
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {e}")

        # ===== Results =====
        st.success(f"✅ {len(matched)} resume(s) match the selected role: **{target_label}**")

        # --- Matching Resumes ---
        st.subheader("🎯 Matching Resumes")
        for name, category, feedback in matched:
            with st.expander(f"✅ {name} — Role: {category}", expanded=False):
                if feedback:
                    try:
                        parsed = eval(feedback)
                        render_feedback(parsed)
                    except:
                        st.text(feedback)

        # --- All Resumes ---
        st.divider()
        st.subheader("📦 All Uploaded Resumes")
        for name, category, feedback in all_results:
            with st.expander(f"📄 {name} — Predicted: {category}"):
                if category == target_label:
                    st.success("🎯 Matched with selected role!")
                    if feedback:
                        try:
                            parsed = eval(feedback)
                            render_feedback(parsed)
                        except:
                            st.text(feedback)
                else:
                    st.info("🚫 Not a match for selected role.")


