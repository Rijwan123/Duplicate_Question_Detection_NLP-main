import os
import gzip
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import helper


st.set_page_config(page_title="Duplicate Question Detector", page_icon="❓")
st.header("Duplicate Question Pairs")

MODEL_FILENAME = "model_quora_duplicate_question_detection_1.pkl"  # or .pkl.gz

def _open_pickle(path: Path):
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")

def _extract_estimator(obj):
    # Already a predictor/pipeline?
    if hasattr(obj, "predict"):
        return obj
    # Dict wrapper? Try common keys.
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "clf", "estimator"):
            est = obj.get(key)
            if hasattr(est, "predict"):
                return est
    return None

@st.cache_resource(show_spinner=True)
def load_predictor(filename: str):
    path = (Path(__file__).resolve().parent / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    with _open_pickle(path) as f:
        obj = pickle.load(f)

    est = _extract_estimator(obj)
    if est is None:
        details = f"dict keys: {list(obj.keys())}" if isinstance(obj, dict) else f"type: {type(obj).__name__}"
        raise TypeError(
            "Loaded object is not a predictor with .predict(). "
            "Expected an sklearn estimator/pipeline, or a dict containing one under keys "
            "['pipeline', 'model', 'clf', 'estimator']. "
            f"Loaded object summary: {details}"
        )
    return est

try:
    model = load_predictor(MODEL_FILENAME)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

q1 = st.text_input("Enter question 1")
q2 = st.text_input("Enter question 2")

def _ensure_2d(x):
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    if not hasattr(x, "ndim"):
        x = np.asarray([x])
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x

if st.button("Find"):
    try:
        query = helper.query_point_creator(q1, q2)
        query = _ensure_2d(query)

        pred = model.predict(query)
        result = bool(pred[0])

        st.header("Duplicate" if result else "Not Duplicate")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(query)[0]
            if len(proba) == 2:
                st.caption(f"Confidence (duplicate): {proba[1]:.3f}")
        elif hasattr(model, "decision_function"):
            df = model.decision_function(query)
            st.caption(f"Decision function: {float(df[0]):.3f}")

    except Exception as e:
        st.error(f"Inference failed: {e}")




# import streamlit as st
# import helper
# import pickle
# import os
# import gzip


# # Get the path of the current file
# current_file = os.path.abspath(__file__)

# # Get the directory containing the current file
# current_dir = os.path.dirname(current_file)

# # Create a path to a new file relative to this script
# file_path = os.path.join(current_dir, "model_quora_duplicate_question_detection_1.pkl")

# print(file_path)


# # try:
# #     with gzip.open(file_path, "rb") as f:
# #         model = pickle.load(f)
# #     st.success("Model loaded successfully ✅")
# # except FileNotFoundError:
# #     st.error(f"Model file not found: {file_path}")
# # except Exception as e:
# #     st.error(f"Error loading model: {e}")


# model = pickle.load(open(file_path,'rb'))

# st.header('Duplicate Question Pairs')

# q1 = st.text_input('Enter question 1')
# q2 = st.text_input('Enter question 2')

# if st.button('Find'):
#     query = helper.query_point_creator(q1,q2)
#     result = model.predict(query)[0]

#     if result:
#         st.header('Duplicate')
#     else:

#         st.header('Not Duplicate')







