import streamlit as st
import helper
import pickle
import os
import gzip


# Get the path of the current file
current_file = os.path.abspath(__file__)

# Get the directory containing the current file
current_dir = os.path.dirname(current_file)

# Create a path to a new file relative to this script
file_path = os.path.join(current_dir, "model_quora_duplicate_question_detection_1.pkl")

print(file_path)


# try:
#     with gzip.open(file_path, "rb") as f:
#         model = pickle.load(f)
#     st.success("Model loaded successfully ✅")
# except FileNotFoundError:
#     st.error(f"Model file not found: {file_path}")
# except Exception as e:
#     st.error(f"Error loading model: {e}")


model = pickle.load(open(new_file,'rb'))

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





