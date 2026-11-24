import os, re, json, joblib
import streamlit as st
import fitz
import torch
from transformers import pipeline

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")
ART="artifacts"
DEVICE=0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_classifier():
    p=os.path.join(ART,"best_classifier.joblib")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data
def load_metadata():
    p=os.path.join(ART,"model_metadata.json")
    if os.path.exists(p):
        with open(p,encoding="utf-8") as f: return json.load(f)
    return {}

@st.cache_resource
def load_summarizer():
    return pipeline("summarization",model="sshleifer/distilbart-cnn-12-6",device=DEVICE)

@st.cache_resource
def load_generator():
    return pipeline("text2text-generation",model="google/flan-t5-small",device=DEVICE)

def clean_text(t):
    t=str(t).lower()
    t=re.sub(r"http\S+|www\\.\S+"," ",t)
    t=re.sub(r"[^a-z0-9\s]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def extract_text(file):
    if file is None: return ""
    raw=file.getvalue()
    if file.name.lower().endswith(".pdf"):
        doc=fitz.open(stream=raw,filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return raw.decode("utf-8",errors="ignore")

def summarize(text,max_len,min_len):
    model=load_summarizer()
    words=text.split(); chunks=[]; current=[]
    for word in words:
        current.append(word)
        if len(current)>=550:
            chunks.append(" ".join(current)); current=[]
    if current: chunks.append(" ".join(current))
    partial=[]
    for chunk in chunks[:10]:
        partial.append(model(chunk,max_length=int(max_len),min_length=int(min_len),
                             do_sample=False)[0]["summary_text"])
    if len(partial)>1:
        return model(" ".join(partial),max_length=int(max_len),
                     min_length=int(min_len),do_sample=False)[0]["summary_text"]
    return partial[0] if partial else ""

def make_aids(text,kind):
    model=load_generator()
    instruction=("Create 6 concise study flashcards. Format each as Q: question and A: answer."
                 if kind=="Flashcards" else
                 "Create 5 factual multiple-choice questions with four options A, B, C and D. Mark the correct answer.")
    prompt=instruction+"\n\nStudy material:\n"+text[:2500]
    return model(prompt,max_new_tokens=350,do_sample=False)[0]["generated_text"]

st.markdown("""
<style>
.block-container {padding-top:2rem;max-width:1100px}
.hero {padding:1.4rem;border-radius:18px;background:linear-gradient(110deg,#182848,#4b6cb7);color:white}
.hero h1 {font-size:2.5rem;margin:0}
.hero p {font-size:1.05rem;margin:.4rem 0 0 0;color:#e7ecff}
</style>
""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>📚 StudyMate AI</h1><p>Upload your notes. Understand faster. Study smarter.</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.title("StudyMate AI")
    st.caption("Machine Learning + Transformer-powered study assistant")
    st.markdown("---")
    st.markdown("**Tools**")
    st.markdown("- PDF/TXT extraction\n- Topic classification\n- AI summarization\n- Flashcards\n- Practice MCQs")
    st.info("AI-generated material may contain mistakes. Verify it against your original notes.")

st.subheader("1. Upload your study material")
uploaded=st.file_uploader("Choose a PDF or TXT file",type=["pdf","txt"])
default=""
if uploaded:
    try:
        default=extract_text(uploaded)
        st.success(f"Extracted {len(default):,} characters from {uploaded.name}.")
    except Exception as e:
        st.error(f"File extraction failed: {e}")
notes=st.text_area("Lecture notes",value=default,height=250,
                   placeholder="Upload a document or paste your lecture notes here...")
if notes.strip(): st.caption(f"{len(notes.split()):,} words · {len(notes):,} characters")

st.subheader("2. Choose a StudyMate tool")
tab1,tab2,tab3=st.tabs(["🧠 Topic Classification","✨ Smart Summary","🗂️ Flashcards & MCQs"])

with tab1:
    st.write("Predict the broad topic using your trained TF-IDF + ML classifier.")
    st.caption("Current AG News categories: World, Sports, Business and Sci/Tech. These are broad topics, not university course labels.")
    if st.button("Predict topic",type="primary"):
        if len(notes.strip())<20: st.warning("Please enter or upload some text.")
        else:
            try:
                clf=load_classifier()
                if clf is None: st.error("Model artifacts are missing. Upload the trained files into the artifacts/ folder in your GitHub repository.")
                else:
                    labels=load_metadata().get("labels",["World","Sports","Business","Sci/Tech"])
                    pred=int(clf.predict([clean_text(notes)])[0])
                    st.success(f"Predicted topic: {labels[pred]}")
            except Exception as e: st.error(f"Classification failed: {e}")

with tab2:
    st.write("Generate a concise summary using the pretrained DistilBART Transformer.")
    c1,c2=st.columns(2)
    max_len=c1.slider("Maximum summary length",60,180,120,10)
    min_len=c2.slider("Minimum summary length",15,80,35,5)
    if st.button("Generate summary",type="primary"):
        if len(notes.strip())<80: st.warning("Please provide a few sentences of text.")
        else:
            try:
                with st.spinner("DistilBART is summarizing your notes. First run may take a few minutes..."):
                    result=summarize(notes,max_len,min_len)
                st.markdown("#### Your summary")
                st.write(result)
                st.download_button("Download summary",result,"studymate_summary.txt")
            except Exception as e: st.error(f"Summarization failed: {e}")

with tab3:
    st.write("Create revision resources using the pretrained FLAN-T5-small model.")
    kind=st.radio("Choose resource",["Flashcards","MCQs"],horizontal=True)
    if st.button("Generate study material",type="primary"):
        if len(notes.strip())<80: st.warning("Please provide a few sentences of text.")
        else:
            try:
                with st.spinner("FLAN-T5 is preparing your study material..."):
                    result=make_aids(notes,kind)
                st.markdown(f"#### Generated {kind}")
                st.write(result)
                st.download_button(f"Download {kind}",result,f"studymate_{kind.lower()}.txt")
            except Exception as e: st.error(f"Generation failed: {e}")

st.markdown("---")
st.caption("StudyMate AI · TF-IDF + classical ML · DistilBART · FLAN-T5-small")
