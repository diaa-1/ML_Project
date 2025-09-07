import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy.sparse import hstack, csr_matrix

# ✅ Load model & vectorizer
model = joblib.load("best_toxic_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ✅ Define labels
labels = ["toxic"]

# ✅ Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"@\w+", "MENTION", text)
    text = re.sub(r"#\w+", "HASHTAG", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ✅ Extract extra features
def extract_features(text):
    return {
        "text_length": len(text),
        "num_words": len(text.split()),
        "num_exclamations": text.count("!"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(1, len(text)),
    }

# ✅ History
if "history" not in st.session_state:
    st.session_state.history = []

# ✅ UI
st.set_page_config(page_title="Content Moderation System", page_icon="🛡️", layout="centered")
st.title("🛡️ Toxic Comment Classification")

user_input = st.text_area("💬 Write a comment:", height=120)

if st.button("Analyze Comment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment first.")
    else:
        clean_text = preprocess_text(user_input)

        # ---- Vectorize ----
        X_vec = vectorizer.transform([clean_text])

        # ---- Extra Features ----
        feats = list(extract_features(clean_text).values())
        extra_feats = csr_matrix([feats])

        # ---- Merge ----
        X_vec_full = hstack([X_vec, extra_feats])
        # ✅ Fix mismatch: pad features if needed
        expected_n_features = model.n_features_in_
        current_n_features = X_vec_full.shape[1]

        if current_n_features < expected_n_features:
                 diff = expected_n_features - current_n_features
                 padding = csr_matrix((X_vec_full.shape[0], diff))
                 X_vec_full = hstack([X_vec_full, padding])
        elif current_n_features > expected_n_features:
            X_vec_full = X_vec_full[:, :expected_n_features]
        

        # ✅ Predict probabilities
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_vec_full)[:, 1]
        else:
            y_probs = model.decision_function(X_vec_full)
            y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())

        # ✅ Display results
        st.subheader("📊 Prediction Results")
        for label, prob in zip(labels, y_probs):
            if prob >= 0.5:
                st.error(f"⚠️ {label.upper()} → {prob:.2f}")
            else:
                st.success(f"✅ {label.upper()} → {prob:.2f}")

        decision = "SUSPICIOUS / TOXIC" if any(prob >= 0.5 for prob in y_probs) else "SAFE"
        st.markdown(f"### {'🔴' if decision!='SAFE' else '🟢'} Final Decision: **{decision}**")

        # ✅ Visualization
        fig, ax = plt.subplots()
        ax.bar(labels, y_probs, color=["red" if p >= 0.5 else "green" for p in y_probs])
        ax.set_ylabel("Probability")
        ax.set_title("Toxicity Level")
        st.pyplot(fig)

        # ✅ Save to history
        entry = {"Comment": user_input, "Decision": decision, "Toxic Probability": f"{y_probs[0]:.2f}"}
        st.session_state.history.append(entry)

# ✅ Show history
if st.session_state.history:
    st.subheader("📝 History Log")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    csv_buffer = io.StringIO()
    hist_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download History", csv_buffer.getvalue(), "history.csv", "text/csv")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("✅ History cleared successfully!")
       
