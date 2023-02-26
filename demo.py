import streamlit as st
import pandas as pd

data_file = st.sidebar.file_uploader('Upload CSV File',type=['csv'])
if data_file is not None:
    df = pd.read_csv(data_file)
    st.dataframe(df.head())