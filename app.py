# app.py
import os
import gzip
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import helper


# ---------------- Page setup ----------------
st.set_page_config(page_title="Duplicate Question Detector", page_icon="❓")
st.header("Duplicate Question Pairs")


# ---------------- Utilities ----------------
def _open_pickle(path: Path):
    """Return a readable file handle for .pkl or .pkl.gz."""
    return gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")

def _has_predict(x):
    return hasattr(x, "predict") and callable(getattr(x, "predict", None))

def _iter_children(x):
    """Yield nested objects to search for a predictor."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, v
    elif isinstance(x, (list, tuple, set)):
        for i, v in enumerate(x):
            yield i, v
    else:
        # common attributes used to wrap estimators
        for attr in ("pipeline", "model", "clf", "estimator", "predictor", "wrapped", "value"):
            if hasattr(x, attr):
                yield attr, getattr(x, attr)

def _find_predictor(obj, _seen=None, _depth=0, _max_depth=6):
    """Recursively search nested containers for an object with .predict()."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return None
    _seen.add(oid)

    if _has_predict(obj):
        return obj
    if _depth >= _max_depth:
        return None

    # Fast-path: unwrap single-key dicts like {'key': actual_estimator}
    if isinstance(obj, dict) and len(obj) == 1:
        (_, only_val), = obj.items()
        found = _find_predictor(only_val, _seen, _depth + 1, _max_depth)
        if found is not None:
            return found

    # Otherwise walk all children
    for _, child in _iter_children(obj):
        try:
            found = _find_predictor(child, _seen, _depth + 1, _max_depth)
            if found is not None:
                return found
        except Exception:
            continue
    return None

def _ensure_2d(x):
    """Ensure feature vector is shape (1, n_features)."""
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    if not hasattr(x, "ndim"):
        x = np.asarray([x])
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


# ---------------- Model loading ----------------
@st.cache_resource(show_spinner=True)
def load_predictor():
    """Try both .pkl and .pkl.gz in the app folder; return an object with .predict()."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "model_quora_duplicate_question_detection_1.pkl",
        here / "model_quora_duplicate_question_detection_1.pkl.gz",
    ]

    paths = [p for p in candidates if p.exists()]
    if not paths:
        raise FileNotFoundError(
            "Model file not found. Expected one of:\n"
            f" - {candidates[0]}\n"
            f" - {candidates[1]}"
        )

    path = paths[0]  # first match wins
    with _open_pickle(path) as f:
        obj = pickle.load(f)

    est = _find_predictor(obj)
    if est is None:
        summary = type(obj).__name__
        if isinstance(obj, dict):
            summary += f" keys={list(obj.keys())[:10]}"
        raise TypeError(
            "Loaded object is not (or does not contain) a predictor with .predict().\n"
            "Please export a single sklearn Pipeline/estimator or a dict that contains one.\n"
            f"Top-level object: {summary}"
        )
    return est


# ---------------- UI + Inference ----------------
try:
    model = load_predictor()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

q1 = st.text_input("Enter question 1")
q2 = st.text_input("Enter question 2")

if st.button("Find"):
    if not q1 or not q2:
        st.warning("Please enter both questions.")
    else:
        try:
            # Build features using your helper
            features = helper.query_point_creator(q1, q2)
            features = _ensure_2d(features)

            pred = model.predict(features)
            result = bool(pred[0])

            st.header("✅ Duplicate" if result else "❌ Not Duplicate")

            # Optional confidence/score
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    if len(proba) == 2:
                        st.caption(f"Confidence (duplicate): {proba[1]:.3f}")
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(features)
                    st.caption(f"Decision function: {float(df[0]):.3f}")
            except Exception:
                pass

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








