import streamlit as st
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer
import torch
import html
from bs4 import BeautifulSoup
import warnings
import time
import faiss

warnings.filterwarnings("ignore")

# running on cpu
device = "cpu"

# load dataset with embeddings
@st.cache_resource
def load_embedding_dataset():
    dataset = load_from_disk('embedding')
    dataset.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)
    return dataset

# load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    checkpoint = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, device_map=device, torch_dtype="auto")
    return model, tokenizer

def html_to_text(html_string):
    unescaped = html.unescape(html_string)
    soup = BeautifulSoup(unescaped, 'html.parser')
    return soup.get_text()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embedding(documents, return_numpy=True):
    inputs = tokenizer(
        documents, padding=True, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_output = model(**inputs)
    embedding = cls_pooling(model_output)
    return embedding.detach().cpu().numpy() if return_numpy else embedding
    
st.set_page_config(page_title="Legal QA Engine", layout="wide")

def find_nearest(query: str, k_nearest: int=5):
    query_embeddings = get_embedding(query, return_numpy=True)
    scores, documents = embedding_dataset.get_nearest_examples("embedding", query_embeddings, k=k_nearest)
    df = pd.DataFrame.from_dict(documents)
    df['faiss_scores'] = scores
    df.sort_values('faiss_scores', ascending=False, inplace=True)
    return df

# load resources
embedding_dataset = load_embedding_dataset()
model, tokenizer = load_model_and_tokenizer()


st.title("🔍 Legal QA Engine")
st.markdown("Search through 38,000+ legal Q&A pairs from Law StackExchange")

query = st.text_input("Enter your legal question:", placeholder="e.g. What constitutes self-defense?")

if st.button("Search") or query:
    start_time = time.time()
    results = find_nearest(query, k_nearest=5).to_dict('records')
    def to_plain_text(field):
        if isinstance(field, dict):
            for k in field:
                field[k] = to_plain_text(field[k])
        return html_to_text(field) if isinstance(field, str) else field
    results = [{k: to_plain_text(v) for k, v in record.items()} for record in results]
    response_time = time.time() - start_time
    # show summary
    st.success(f"Found {len(results)} results in {response_time:.2f} seconds")
    
    # display results
    for i, result in enumerate(results):
        with st.expander(f"#{i + 1}: {result['question_title']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(result['question_title'])
                st.caption(f"🔗 [Source]({result['link']})")
                st.write(f"🏷️ Tags: {result['tags']}")
                st.write(f"⬆️ Question Score: {result['question_score']} | 🤗 Answer Score: {result['answer_score']}")
                
                st.markdown("**Question:**")
                st.write(result['question_body'])
                
                st.markdown("**Answer:**")
                st.write(result['answers']['body'])
            
            with col2:
                st.metric("Similarity Score", f"{result['faiss_scores']:.2f}")
                st.progress(min(float(result['faiss_scores']/30), 1.0))