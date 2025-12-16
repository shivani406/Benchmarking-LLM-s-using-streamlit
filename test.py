import google.generativeai as genai
import streamlit as st
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

models = list(genai.list_models())
print(models)
