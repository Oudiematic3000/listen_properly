import os
import json
import time
import random
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types
import pandas as pd

st.set_page_config(page_title="isiZulu AI Evaluation", layout="wide")
st.title("isiZulu Neurosymbolic AI Evaluation Suite")

tab1, tab2 = st.tabs(["Experiment 1: Native Reasoning (Gemini)", "Experiment 2: Fine-Tuned Adapters (MzansiLM)"])

# --- GEMINI CALL WITH FAIL-FAST RETRY ---
def call_gemini_with_retry(client, prompt_text, max_retries=3):
    model_name = "gemini-3.6-flash"
    delay = 2  
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(temperature=0.3) # Slight temp for natural generation
            )
            return (response.text or "").strip()
        except Exception as e:
            err_msg = str(e).lower()
            if any(err in err_msg for err in ["503", "busy", "quota", "429", "rate limit", "exhausted"]):
                time.sleep(delay)
                delay *= 2 
                continue
            return f"[API Error]: {str(e)}"
            
    return "[Timeout]: Exceeded retries waiting for Gemini API."


# --- TAB 1: NATIVE ISIZULU REASONING EXPERIMENT ---
def format_morphology(tokens):
    # Formats structural hints if provided in the dataset
    if not tokens: return ""
    lines = [f"  {t.get('word', '?')} [{t.get('pos_tag', '?')}] root={t.get('morphology', {}).get('root', '?')}" for t in tokens]
    return "\n".join(lines)

def build_qa_prompt(condition, question, context, fewshot_pool):
    instruction = (
        "Wena ungumsizi okhaliphile wesiZulu. Phendula umbuzo olandelayo ngokusebenzisa "
        "ulwazi olunikeziwe. Phendula ngesiZulu kuphela, ube mfushane futhi ucacise.\n\n"
    )
    
    # Zero Shot
    if condition == "zero_shot":
        return f"{instruction}Ulwazi (Context): {context}\nUmbuzo (Question): {question}\nImpendulo:"
    
    # Few Shot (Plain or Annotated)
    blocks = []
    for ex in fewshot_pool:
        q = ex.get('question', '')
        c = ex.get('context', '')
        a = ex.get('answer', '')
        
        if condition == "fewshot_plain":
            blocks.append(f"Ulwazi: {c}\nUmbuzo: {q}\nImpendulo: {a}")
        elif condition == "fewshot_annotated":
            morph = format_morphology(ex.get('tokens', []))
            morph_str = f"Ukwakheka kwamagama (Morphology):\n{morph}\n" if morph else ""
            blocks.append(f"Ulwazi: {c}\nUmbuzo: {q}\n{morph_str}Impendulo: {a}")
            
    prefix = "\n\n".join(blocks)
    return f"{instruction}{prefix}\n\nUlwazi: {context}\nUmbuzo: {question}\nImpendulo:"

with tab1:
    st.header("Native isiZulu Reasoning & QA (Gemini 3.6 Flash)")
    st.markdown("Tests if the model can read, reason, and answer **purely in isiZulu** without relying on translation. Evaluates if morphological scaffolding improves concordial agreement and logic.")
    
    col_a, col_b, col_c = st.columns(3)
    dataset_path = col_a.text_input("Dataset Path", "isizulu_qa_dataset.json")
    fewshot_k = col_b.number_input("Few-Shot Examples (K)", min_value=0, max_value=5, value=2)
    test_size = col_c.number_input("Test Set Size", min_value=1, max_value=20, value=3)
    
    if st.button("Run Native QA Experiment"):
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        elif not os.path.exists(dataset_path):
            st.error(f"Could not find dataset at {dataset_path}. (Ensure your JSON has 'context', 'question', and 'answer' keys).")
        else:
            client = genai.Client(api_key=api_key)
            
            with open(dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # Filter for items that actually have QA pairs
            data = [d for d in raw_data if d.get("question") and d.get("context")]
            random.seed(42)
            random.shuffle(data)
            
            fewshot_pool = data[:fewshot_k]
            test_pool = data[fewshot_k:fewshot_k + test_size]
            
            # Dynamic UI Elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create an empty container to hold the live-updating dataframe
            table_container = st.empty()
            live_results = []
            
            conditions = ["zero_shot", "fewshot_plain", "fewshot_annotated"]
            
            for i, item in enumerate(test_pool):
                question = item['question']
                context = item['context']
                
                row_data = {
                    "Context (Ulwazi)": context[:100] + "...", # Truncated for UI neatness
                    "Question (Umbuzo)": question,
                }
                
                for cond in conditions:
                    status_text.text(f"Processing item {i+1}/{len(test_pool)} | Condition: {cond.replace('_', ' ').title()}")
                    prompt = build_qa_prompt(cond, question, context, fewshot_pool)
                    
                    response_text = call_gemini_with_retry(client, prompt)
                    row_data[cond.replace("_", " ").title()] = response_text
                    time.sleep(1.5) # Gentle rate-limit pause
                
                # Append to live results and update the UI instantly
                live_results.append(row_data)
                df = pd.DataFrame(live_results)
                table_container.dataframe(df, use_container_width=True)
                
                progress_bar.progress((i + 1) / len(test_pool))
            
            status_text.text("✅ Evaluation Complete!")

# --- MZANSILM CACHED MODEL LOAD (Unchanged) ---
@st.cache_resource
def load_mzansilm():
    MODEL_ID = "uctnlp/mzansilm-125m"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
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