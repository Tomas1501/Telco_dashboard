import streamlit as st

hide_toolbar = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_toolbar, unsafe_allow_html=True)
st.set_page_config(
    page_title="Telco dashboard",
    # page_icon="📊",
    layout="wide",
    # initial_sidebar_state="expanded"
)

st.title('Telco dashboard')