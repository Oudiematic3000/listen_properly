import os, json, time
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai

st.set_page_config(page_title="isiZulu AI Evaluation", layout="wide")
st.title("isiZulu Neurosymbolic AI Evaluation Suite")

tab1, tab2 = st.tabs(["Experiment 1: In-Context Prompting (Gemini)", "Experiment 2: Fine-Tuned Adapters (MzansiLM)"])

# --- GEMINI CALL WITH MODEL FALLBACKS ---
def call_gemini_with_fallback(client, prompt_text):
    # Primary model first, followed by secondary fallbacks
    fallback_models = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    for model_name in fallback_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text
                )
                return response.text
            except Exception as e:
                # Catch rate limits, 503 service unavailable, or quota limits
                if any(err in str(e).lower() for err in ["503", "busy", "quota", "429"]):
                    time.sleep(1)
                    continue
                raise e
    raise RuntimeError("All Gemini endpoints are currently busy. Please try again shortly.")

with tab1:
    st.header("In-Context Symbolic Scaffolding (Gemini)")
    user_prompt = st.text_input("isiZulu Prompt", "Chaza ukuthi yini ubulungiswa esizweni?", key="g_prompt")
    morph_input = st.text_input("Morphology Context", "ubulungiswa: class 14 (ubu-), root: lungisa", key="g_morph")
    
    if st.button("Run Gemini Evaluation"):
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        else:
            client = genai.Client(api_key=api_key)
            with st.spinner("Querying Gemini..."):
                try:
                    res_a = call_gemini_with_fallback(client, f"Answer the following isiZulu prompt directly: {user_prompt}")
                    res_b = call_gemini_with_fallback(client, f"Linguistic context:\n{morph_input}\n\nUsing this morphology, answer: {user_prompt}")
                    res_c = call_gemini_with_fallback(client, f"Step 1: Identify isiZulu word prefixes/roots/noun classes.\nStep 2: Generate response adhering to concordial rules.\n\nPrompt: {user_prompt}")

                    col1, col2, col3 = st.columns(3)
                    col1.subheader("Condition A (Direct)")
                    col1.write(res_a)
                    col2.subheader("Condition B (External Scaffold)")
                    col2.write(res_b)
                    col3.subheader("Condition C (Symbolic CoT)")
                    col3.write(res_c)
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- MZANSILM CACHED MODEL LOAD ---
@st.cache_resource
def load_mzansilm():
    MODEL_ID = "uctnlp/mzansilm-125m"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Low memory flags for 1GB RAM Streamlit Cloud instances
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