
import os, tempfile, pandas as pd, streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup

EMBED = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

def df_to_docs(df, tag):
    return [Document(page_content=row.to_json(), metadata={"where": tag})
            for _, row in df.iterrows()]

def html_to_docs(html_bytes):
    soup = BeautifulSoup(html_bytes, "html.parser")
    text = "\n".join(t.get_text(" ", strip=True) for t in soup.select("body *")
                      if t.get_text(strip=True))
    return SPLITTER.split_documents([Document(page_content=text, metadata={"where": "facebook"})])

def build_chain(docs):
    if not docs:
        return None
    store = FAISS.from_documents(docs, EMBED)
    return RetrievalQA.from_chain_type(
        llm = ChatOpenAI(
            model = os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature = 0.2,
            max_tokens = 2048,
        ),
        chain_type = "stuff",
        retriever = store.as_retriever(),
        return_source_documents = True,
    )

# ---------------- Streamlit ----------------
st.set_page_config(page_title="Spine Copilot", page_icon="🏃‍♂️")
st.title("🏔️ Spine Copilot – single‑file")

with st.sidebar:
    st.header("Upload")
    res_csv = st.file_uploader("Race results CSV (required)", type="csv")
    rec_csv = st.file_uploader("Course records CSV (optional)", type="csv")
    fb_html = st.file_uploader("Facebook gallery HTML (optional)", type="html")
    use_fb  = st.checkbox("Include FB data", value=False)

if res_csv is None:
    st.info("Please upload the race-results CSV to start.")
    st.stop()

@st.cache_data(show_spinner="Parsing CSVs…")
def read_csv(upload):
    return (pd.read_csv(upload)
              .rename(columns=lambda c: c.strip().lower().replace(" ", "_")))

res_df = read_csv(res_csv)
rec_df = read_csv(rec_csv) if rec_csv else pd.DataFrame()

docs = df_to_docs(res_df, "results") + df_to_docs(rec_df, "records")
main_chain = build_chain(docs)

fb_chain = None
if use_fb and fb_html is not None:
    fb_docs = html_to_docs(fb_html.read())
    fb_chain = build_chain(fb_docs)

prompt = st.text_input("Ask a question…")
if prompt:
    answer, sources = "", []
    if main_chain:
        out = main_chain({"query": prompt})
        answer, sources = out["result"], out["source_documents"]
    if fb_chain:
        fb_out = fb_chain({"query": prompt})
        answer += "\n\n---\n\n" + fb_out["result"]
        sources += fb_out["source_documents"]
    st.markdown(answer)
    if sources:
        with st.expander("Sources"):
            for doc in sources:
                st.write(doc.page_content[:400])
