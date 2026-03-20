import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Paths
MODEL_PATH  = "autocorrect_model.h5"
VOCAB_PATH  = "vocab.pkl"
CONFIG_PATH = "config.pkl"

@st.cache_resource
def load_all():
    # ✅ Fix 1: use compile=False to avoid h5py/keras version conflicts
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # ✅ Fix 2: vocab.pkl was saved as a (vocab, inv_vocab) TUPLE
    with open(VOCAB_PATH, "rb") as f:
        char_to_idx, idx_to_char = pickle.load(f)

    # ✅ Fix 3: config.pkl only has max_len and vocab_size
    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    max_len = config["max_len"]

    return model, char_to_idx, idx_to_char, max_len

model, char_to_idx, idx_to_char, max_len = load_all()

# Encoding function
def encode_word(word):
    seq = [char_to_idx.get(c, 0) for c in word.lower()]
    seq = seq[:max_len]
    seq += [0] * (max_len - len(seq))
    return np.array(seq)

# Decoding function
def decode_output(pred_indices):
    special = {"<PAD>", "<SOS>", "<EOS>", ""}
    return "".join(
        idx_to_char.get(i, "")
        for i in pred_indices
        if idx_to_char.get(i, "") not in special
    )

# Prediction function
def predict_word(word):
    seq  = encode_word(word)
    seq  = np.expand_dims(seq, axis=0)          # shape: (1, max_len)
    preds = model.predict(seq, verbose=0)        # shape: (1, max_len, vocab_size)
    pred_indices = np.argmax(preds, axis=-1)[0]  # shape: (max_len,)
    return decode_output(pred_indices)

# Streamlit UI
st.title("🔤 Autocorrect LSTM App")
st.write("Enter a misspelled word:")

user_input = st.text_input("Input word")

if st.button("Correct"):
    if user_input.strip() == "":
        st.warning("Please enter a word.")
    else:
        corrected = predict_word(user_input.lower())
        st.success(f"Corrected word: **{corrected}**")