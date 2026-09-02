import os
import json
import time
import random
import streamlit as st
import torch
import sacrebleu
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from google import genai
from google.genai import types

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
                    contents=prompt_text,
                    # Force temperature to 0 for translation evaluations
                    config=types.GenerateContentConfig(temperature=0.0) 
                )
                return (response.text or "").strip()
            except Exception as e:
                # Catch rate limits, 503 service unavailable, or quota limits
                if any(err in str(e).lower() for err in ["503", "busy", "quota", "429"]):
                    time.sleep(2)
                    continue
                raise e
    raise RuntimeError("All Gemini endpoints are currently busy. Please try again shortly.")


# --- TAB 1: IN-CONTEXT LEARNING EXPERIMENT ---
def format_morphology(tokens):
    lines = []
    for t in tokens:
        morph = t.get("morphology", {}) or {}
        lines.append(
            f"  {t.get('word', '?')} [{t.get('pos_tag', '?')}] "
            f"root={morph.get('root', '?')} prefix={morph.get('prefix', '?')}"
        )
    return "\n".join(lines)

def build_prompt(condition, english_sentence, fewshot_pool):
    instruction = (
        "You are translating English into isiZulu. Study the reference "
        "examples below, then translate the final sentence. Respond with "
        "ONLY the isiZulu translation and nothing else — no explanation, "
        "no English.\n\n"
    )
    if condition == "zero_shot":
        return f"{instruction}English: {english_sentence}\nisiZulu:"
    
    blocks = []
    for ex in fewshot_pool:
        if condition == "fewshot_plain":
            blocks.append(f"English: {ex['english_translation']}\nisiZulu: {ex['original_sentence']}")
        elif condition == "fewshot_annotated":
            blocks.append(
                f"English: {ex['english_translation']}\n"
                f"isiZulu: {ex['original_sentence']}\n"
                f"Morphology:\n{format_morphology(ex.get('tokens', []))}"
            )
    
    prefix = "\n\n".join(blocks)
    return f"{instruction}{prefix}\n\nEnglish: {english_sentence}\nisiZulu:"

with tab1:
    st.header("In-Context Translation Evaluation (Gemini)")
    st.markdown("Evaluates if morphological annotations improve few-shot translation vs. plain examples.")
    
    col_a, col_b, col_c = st.columns(3)
    dataset_path = col_a.text_input("Dataset Path", "neurosymbolic_dataset.json")
    fewshot_k = col_b.number_input("Few-Shot Examples (K)", min_value=1, max_value=15, value=8)
    test_size = col_c.number_input("Test Set Size", min_value=1, max_value=50, value=5)
    
    if st.button("Run ICL Experiment"):
        api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
        elif not os.path.exists(dataset_path):
            st.error(f"Could not find dataset at {dataset_path}. Please upload it to your repository.")
        else:
            client = genai.Client(api_key=api_key)
            
            # Load and Split Data
            with open(dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            data = [d for d in raw_data if d.get("original_sentence") and d.get("english_translation")]
            random.seed(42)
            random.shuffle(data)
            
            needed = fewshot_k + test_size
            if len(data) < needed:
                st.warning(f"Dataset only has {len(data)} valid items. Shrinking test size.")
                test_size = max(1, len(data) - fewshot_k)
                
            fewshot_pool = data[:fewshot_k]
            test_pool = data[fewshot_k:fewshot_k + test_size]
            
            conditions = ["zero_shot", "fewshot_plain", "fewshot_annotated"]
            results = {c: [] for c in conditions}
            references = [item["original_sentence"] for item in test_pool]
            
            # UI Elements for live updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            # Run the generations
            for i, item in enumerate(test_pool):
                eng_text = item['english_translation']
                status_text.text(f"Translating {i+1}/{len(test_pool)}: {eng_text}")
                
                for cond in conditions:
                    prompt = build_prompt(cond, eng_text, fewshot_pool)
                    try:
                        out_text = call_gemini_with_fallback(client, prompt)
                    except Exception as e:
                        out_text = f"[ERROR: {e}]"
                    results[cond].append(out_text)
                    time.sleep(1) # Gentle rate-limit pause
                
                progress_bar.progress((i + 1) / len(test_pool))
            
            status_text.text("Scoring complete!")
            
            # Calculate Scores
            st.subheader("Evaluation Scores")
            score_cols = st.columns(3)
            for i, cond in enumerate(conditions):
                hyps = results[cond]
                chrf = sacrebleu.corpus_chrf(hyps, [references]).score
                bleu = sacrebleu.corpus_bleu(hyps, [references]).score
                
                with score_cols[i]:
                    st.markdown(f"**{cond.replace('_', ' ').title()}**")
                    st.metric("chrF Score", f"{chrf:.2f}")
                    st.metric("BLEU Score", f"{bleu:.2f}")
            
            # Show Generation Table
            st.subheader("Generation Results")
            table_data = []
            for i in range(len(test_pool)):
                table_data.append({
                    "English": test_pool[i]["english_translation"],
                    "Ground Truth (isiZulu)": references[i],
                    "Zero Shot": results["zero_shot"][i],
                    "Few-Shot Plain": results["fewshot_plain"][i],
                    "Few-Shot Annotated": results["fewshot_annotated"][i],
                })
            st.dataframe(table_data, use_container_width=True)


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