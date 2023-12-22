import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class_list = {'0': 'Positive', '1': 'Neutral', '2': 'Negative'}

X_train = pd.read_csv('./sents.txt', sep='\r\n', header=None, index_col=None, names = ['sents'])

encoder = CountVectorizer(ngram_range = (1,1))
encoder.fit(X_train['sents'])

input_md = open('lrc_vsfc.pkl', 'rb')
model = pkl.load(input_md)

st.header('write a feedback')
txt = st.text_area('', '')

if txt != '':
     if st.button("predict"):
          feature_vector = encoder.transform([txt])
          label = str((model.predict(feature_vector))[0])

          st.header('Result')
          st.text(class_list[label])