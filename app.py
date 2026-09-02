import os
import json
import time
import requests
import streamlit as st
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types

st.set_page_config(page_title="isiZulu AI Evaluation", layout="wide")
st.title("isiZulu Neurosymbolic AI Evaluation Suite")

tab1, tab2 = st.tabs(["Experiment 1: Fine-Tuned Adapters (MzansiLM)", "Experiment 2: Native Reasoning (Gemini / Qwen Fallback)"])


# --- QWEN FALLBACK FUNCTION (via Groq API) ---
def call_qwen_fallback(prompt_text, status_widget):
    """Fallback generator using Qwen 3.8 (qwen/qwen3.8-27b) when Gemini hits limits."""
    groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if not groq_key:
        return "[Quota Exceeded]: Gemini limit reached, and no GROQ_API_KEY was found in secrets."
    
    status_widget.warning("Gemini rate/quota limit reached. Switching to Qwen fallback (`qwen/qwen3.8-27b`)...")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen3.8-27b",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        else:
            return f"[Qwen Error {response.status_code}]: {response.text}"
    except Exception as e:
        return f"[Qwen Exception]: {str(e)}"


# --- MULTI-KEY RETRIEVAL & ROTATION ---
def get_all_api_keys():
    """Extracts all valid API keys from secrets or environment variables."""
    raw_keys = st.secrets.get("GEMINI_API_KEYS") or st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    keys = []
    
    if isinstance(raw_keys, list):
        for item in raw_keys:
            if isinstance(item, str) and item.strip():
                keys.append(item.strip())
    elif isinstance(raw_keys, str) and raw_keys.strip():
        for item in raw_keys.split(","):
            if item.strip():
                keys.append(item.strip())
                
    unique_keys = []
    for k in keys:
        if k not in unique_keys:
            unique_keys.append(k)
    return unique_keys


def call_gemini_with_key_rotation(prompt_text, status_widget, max_retries_per_key=1):
    keys = get_all_api_keys()
    
    # If no Gemini keys are configured, fallback directly to Qwen
    if not keys:
        return call_qwen_fallback(prompt_text, status_widget)
    
    if "active_key_idx" not in st.session_state:
        st.session_state.active_key_idx = 0
        
    model_name = "gemini-3.6-flash"
    total_attempts = len(keys) * max_retries_per_key
    
    for attempt in range(total_attempts):
        current_idx = st.session_state.active_key_idx % len(keys)
        active_key = keys[current_idx]
        
        try:
            client = genai.Client(api_key=active_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return (response.text or "").strip()
            
        except Exception as e:
            err_msg = str(e)
            
            if any(err in err_msg.lower() for err in ["429", "rate limit", "quota", "resource_exhausted"]):
                # Rotate key index
                st.session_state.active_key_idx = (st.session_state.active_key_idx + 1) % len(keys)
                next_idx = st.session_state.active_key_idx % len(keys)
                
                # Trigger Qwen fallback if all Gemini keys fail
                if attempt == total_attempts - 1:
                    return call_qwen_fallback(prompt_text, status_widget)
                
                status_widget.warning(f"Gemini Key #{current_idx + 1} limited. Trying Key #{next_idx + 1}...")
                time.sleep(0.5)
                continue
                
            elif any(err in err_msg.lower() for err in ["503", "busy"]):
                time.sleep(2)
                continue
                
            return f"[API Error]: {err_msg}"
            
    return call_qwen_fallback(prompt_text, status_widget)


# --- PROMPT BUILDERS ---
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


# --- TAB 2: NATIVE ISIZULU REASONING EXPERIMENT ---
with tab2:
    st.header("Simulated low-resource isiZulu comprehension")
    st.markdown("Simulates fine-tuning with lingustic structure by supplying it in prompt.")
    
    dataset_path = st.text_input("Dataset Path", "isizulu_qa_dataset.json")
    
    dataset_items = []
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            dataset_items = [d for d in raw_data if d.get("question") and d.get("context")]
        except Exception as e:
            st.error(f"Error reading JSON dataset: {e}")
    else:
        st.error(f"Could not find dataset file at `{dataset_path}`.")

    if dataset_items:
        all_questions = [item["question"] for item in dataset_items]
        
        col_select, col_k = st.columns([2, 1])
        selected_questions = col_select.multiselect(
            "Select Questions to Include in Test Run",
            options=all_questions,
            default=all_questions[:2]
        )
        fewshot_k = col_k.number_input("Few-Shot Pool Size (K)", min_value=0, max_value=5, value=2)
        
        if st.button("Ask"):
            if not selected_questions:
                st.warning("Please select at least one question from the multiselect dropdown.")
            else:
                test_pool = [d for d in dataset_items if d["question"] in selected_questions]
                remaining_pool = [d for d in dataset_items if d["question"] not in selected_questions]
                fewshot_pool = remaining_pool[:fewshot_k] if remaining_pool else dataset_items[:fewshot_k]
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                table_container = st.empty()
                
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
                        status_text.info(f"Processing Question {i+1}/{len(test_pool)} | Condition: {cond.replace('_', ' ').title()}")
                        prompt = build_qa_prompt(cond, question, context, fewshot_pool)
                        
                        response_text = call_gemini_with_key_rotation(prompt, status_text)
                        row_data[cond.replace("_", " ").title()] = response_text
                        
                        time.sleep(0.5)
                    
                    st.session_state.exp_results.append(row_data)
                    table_container.dataframe(pd.DataFrame(st.session_state.exp_results), use_container_width=True)
                    progress_bar.progress((i + 1) / len(test_pool))
                
                status_text.success("Evaluation Complete!")


# --- TAB 2: MZANSILM ADAPTERS ---
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

with tab1:
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