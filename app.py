import streamlit as st
import google.generativeai as genai 
import time
import pandas as pd
import plotly.express as px
from groq import Groq

st.set_page_config(page_title="GHW LLM Benchmarking", layout="wide")
st.title("LLM Benchmarking Dashboard")

st.subheader("Evaluate and Compare Large Language Models")
st.divider()

client = genai.configure(api_key = st.secrets["GEMINI_API_KEY"])
client_groq = Groq(api_key = st.secrets["GROQ_API_KEY"])


def call_gemini(prompt):
    start = time.time()

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    end = time.time()

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens = response.usage_metadata.total_token_count
    else:
        tokens = len(prompt + response.text) // 4  # rough estimate 

    return response.text, end - start, tokens



def call_llama(prompt):
    start = time.time()
    response_groq = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    end = time.time()
    content = response_groq.choices[0].message.content
    token_count = response_groq.usage.total_tokens
    
    return content, end - start, token_count


# design the sidebar
with st.sidebar:
    st.title("Choose your model")
    use_gemini = st.checkbox("Use Gemini-2.5", value=True)
    use_groq = st.checkbox("Use Llama-3.1", value=True)

prompt = st.chat_input("Enter your prompt-")

if prompt:
    comparisions = []
    if use_gemini:
        comparisions.append("Gemini-2.5 Flash")
    if use_groq:
        comparisions.append("Llama-3.1")

    cols = st.columns(len(comparisions))
    results = []

    for i, comparision_name in enumerate(comparisions):
        with cols[i]:
            st.subheader(comparision_name)
            if comparision_name == "Gemini-2.5 Flash":
                content, latency, tokens = call_gemini(prompt)
            else:
                content, latency, tokens = call_llama(prompt)
        st.caption(f"Latency: {latency:.2f} seconds | Tokens: {tokens}")
        st.write(content)

        if latency > 0:
            results.append({
                "Model": comparision_name,
                "Latency (s)": latency,
                "Tokens": tokens,
                "Throughput (tokens/s)": tokens / latency})
    if results:
        df = pd.DataFrame(results)
        st.subheader("Performance Comparison")
        st.dataframe(df)

        fig = px.bar(df, x="Model", y="Throughput (tokens/s)", title="Model Throughput Comparison", text="Throughput (tokens/s)")
        st.plotly_chart(fig, use_container_width=True)
