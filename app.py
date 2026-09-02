import os
import json
import time
import random
import streamlit as st
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types

st.set_page_config(page_title="isiZulu AI Evaluation", layout="wide")
st.title("isiZulu Neurosymbolic AI Evaluation Suite")

tab1, tab2 = st.tabs(["Experiment 1: Native Reasoning (Gemini)", "Experiment 2: Fine-Tuned Adapters (MzansiLM)"])

# --- GEMINI CALL WITH SMART RATE-LIMIT BACKOFF ---
import itertools

# Set GEMINI_API_KEYS as a list in Streamlit secrets: GEMINI_API_KEYS = ["key1", "key2"]
raw_keys = st.secrets.get("GEMINI_API_KEYS", [os.environ.get("GEMINI_API_KEY")])
key_pool = itertools.cycle(raw_keys)

def call_gemini_with_key_rotation(prompt_text, status_widget):
    for _ in range(len(raw_keys)):
        active_key = next(key_pool)
        client = genai.Client(api_key=active_key)
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt_text,
            )
            return (response.text or "").strip()
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                status_widget.warning("Key daily quota reached. Rotating to next API key...")
                continue
            return f"[API Error]: {str(e)}"
    return "[All Keys Exhausted]: Daily quotas reached on all available keys."

# --- TAB 1: NATIVE ISIZULU REASONING EXPERIMENT ---
def format_morphology(tokens):
    if not tokens: return ""
    lines = [f"  {t.get('word', '?')} [{t.get('pos_tag', '?')}] root={t.get('morphology', {}).get('root', '?')}" for t in tokens]
    return "\n".join(lines)

def build_qa_prompt(condition, question, context, fewshot_pool):
    instruction = (
        "Wena ungumsizi okhaliphile wesiZulu. Phendula umbuzo olandelayo ngokusebenzisa "
        "ulwazi olunikeziwe. Phendula ngesiZulu kuphela, ube mfushane futhi ucacise.\n\n"
    )
    
    if condition == "zero_shot":
        return f"{instruction}Ulwazi (Context): {context}\nUmbuzo (Question): {question}\nImpendulo:"
    
    blocks = []
    for ex in fewshot_pool:
        q, c, a = ex.get('question', ''), ex.get('context', ''), ex.get('answer', '')
        if condition == "fewshot_plain":
            blocks.append(f"Ulwazi: {c}\nUmbuzo: {q}\nImpendulo: {a}")
        elif condition == "fewshot_annotated":
            morph = format_morphology(ex.get('tokens', []))
            morph_str = f"Ukwakheka kwamagama (Morphology):\n{morph}\n" if morph else ""
            blocks.append(f"Ulwazi: {c}\nUmbuzo: {q}\n{morph_str}Impendulo: {a}")
            
    prefix = "\n\n".join(blocks)
    return f"{instruction}{prefix}\n\nUlwazi: {context}\nUmbuzo: {question}\nImpendulo:"

# --- TAB 1: NATIVE ISIZULU REASONING EXPERIMENT ---
with tab1:
    st.header("Native isiZulu Reasoning & QA (Gemini 2.5 Flash)")
    st.markdown("Evaluates whether morphological context enhances native isiZulu reading comprehension.")
    
    col_a, col_b, col_c = st.columns(3)
    dataset_path = col_a.text_input("Dataset Path", "isizulu_qa_dataset.json")
    fewshot_k = col_b.number_input("Few-Shot Examples (K)", min_value=0, max_value=5, value=2)
    test_size = col_c.number_input("Test Set Size", min_value=1, max_value=10, value=2)
    
    if st.button("Run Native QA Experiment"):
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        elif not os.path.exists(dataset_path):
            st.error(f"Could not find dataset at {dataset_path}.")
        else:
            client = genai.Client(api_key=api_key)
            
            with open(dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            
            data = [d for d in raw_data if d.get("question") and d.get("context")]
            random.seed(42)
            random.shuffle(data)
            
            fewshot_pool = data[:fewshot_k]
            test_pool = data[fewshot_k:fewshot_k + test_size]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            table_container = st.empty()
            
            # Initialize results state
            if "exp_results" not in st.session_state:
                st.session_state.exp_results = []
                
            conditions = ["zero_shot", "fewshot_plain", "fewshot_annotated"]
            
            for i, item in enumerate(test_pool):
                question = item['question']
                context = item['context']
                
                row_data = {
                    "Context": context[:70] + "...",
                    "Question": question,
                }
                
                for cond in conditions:
                    status_text.info(f"Item {i+1}/{len(test_pool)} | Running {cond.replace('_', ' ').title()}...")
                    prompt = build_qa_prompt(cond, question, context, fewshot_pool)
                    
                    response_text = call_gemini_with_retry(client, prompt, status_text)
                    row_data[cond.replace("_", " ").title()] = response_text
                    
                    # 5-second delay guarantees <12 RPM (below 15 RPM free tier limit)
                    time.sleep(5) 
                
                st.session_state.exp_results.append(row_data)
                table_container.dataframe(pd.DataFrame(st.session_state.exp_results), use_container_width=True)
                progress_bar.progress((i + 1) / len(test_pool))
            
            status_text.success("✅ Evaluation Complete!")

# --- MZANSILM TAB ---
@st.cache_resource
def load_mzansilm():
    MODEL_ID = "uctnlp/mzansilm-125m"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, "Oudiematic3000/mzansilm-125m-baseline-lora", adapter_name="baseline")
    model.load_adapter("Oudiematic3000/mzansilm-125m-neurosymbolic-lora", adapter_name="neuro")
    model.eval()
    return tokenizer, model

with tab2:
    st.header("Fine-Tuned Adapter Comparison (MzansiLM-125M)")
    inp_m = st.text_input("isiZulu Prompt", "Chaza ukuthi yini ubulungiswa esizweni?", key="m_prompt")
    EXAMPLE_MORPH = json.dumps([
        {"token": "Chaza", "root": "chaza", "pos": "VERB"},
        {"token": "ubulungiswa", "root": "lungisa", "pos": "NOUN", "class": "14"},
        {"token": "esizweni", "root": "sizwe", "pos": "NOUN_LOC", "class": "7"}
    ], ensure_ascii=False)
    morph_m = st.text_area("Morphology Graph (JSON)", value=EXAMPLE_MORPH, height=100)
    
    if st.button("Compare Adapters"):
        with st.spinner("Loading model and generating outputs..."):
            try:
                tokenizer, peft_model = load_mzansilm()
                def gen(p_text, adapter):
                    peft_model.set_adapter(adapter)
                    torch.manual_seed(42)
                    inputs = tokenizer(p_text, return_tensors="pt")
                    with torch.no_grad():
                        outputs = peft_model.generate(**inputs, max_new_tokens=60, do_sample=False, repetition_penalty=1.2)
                    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(p_text):].strip()

                b_prompt = f"isiZulu Input: {inp_m}\nisiZulu Response:"
                n_prompt = f"isiZulu Input: {inp_m}\nMorphology Graph: {morph_m.strip()}\nisiZulu Response:" if morph_m.strip() else b_prompt

                col1, col2 = st.columns(2)
                col1.subheader("Baseline Adapter Output")
                col1.write(gen(b_prompt, "baseline"))
                col2.subheader("Neurosymbolic Adapter Output")
                col2.write(gen(n_prompt, "neuro"))
            except Exception as e:
                st.error(f"Model Inference Error: {e}")