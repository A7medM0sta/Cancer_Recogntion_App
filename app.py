import streamlit as st
import pickle5 as pickle
import pandas as pd

def main():
    # print('Hello World we will build a Stream-Lit')
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon='female-doctor',
        layout='wide',
        initial_sidebar_state='expanded')

    st.write("Hello World")

if __name__ == '__main__':
    main()