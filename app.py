import streamlit as st
import helper
import pickle

import os

# Get the path of the current file
current_file = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file)

# Create a path to a new file relative to this script
new_file = os.path.join(current_dir, "model_quora_duplicate_question_detection_1.pkl")

print(new_file)

model = pickle.load(open(r'new_file','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:

        st.header('Not Duplicate')

