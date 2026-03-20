import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
import re

# -------- LOAD EVERYTHING --------
@st.cache_resource
def load_all():
    model = keras.models.load_model("auto.h5", compile=False)

    with open("var.pkl", "rb") as f:
        data = pickle.load(f)

    # Build encoder-decoder
    latent_dim = 128

    encoder = keras.Model(
        model.input[0],
        model.layers[2].output[1:]
    )

    decoder_input = keras.Input(shape=(1, data["num_decoder_tokens"]))
    h = keras.Input(shape=(latent_dim,))
    c = keras.Input(shape=(latent_dim,))

    dec_lstm = model.layers[3]
    dec_dense = model.layers[4]

    out, h_new, c_new = dec_lstm(decoder_input, initial_state=[h, c])
    out = dec_dense(out)

    decoder = keras.Model(
        [decoder_input, h, c],
        [out, h_new, c_new]
    )

    reverse = {i: c for c, i in data["target_token_index"].items()}

    return encoder, decoder, data, reverse


# -------- SIMPLE PREDICT FUNCTION --------
def predict(text, encoder, decoder, data, reverse):
    text = re.sub(r'[^a-z ]', '', text.lower())

    enc = np.zeros((1, data["max_encoder_seq_length"], data["num_encoder_tokens"]))

    for t, ch in enumerate(text):
        if ch in data["input_token_index"]:
            enc[0, t, data["input_token_index"][ch]] = 1

    states = encoder.predict(enc, verbose=0)

    target = np.zeros((1, 1, data["num_decoder_tokens"]))
    target[0, 0, data["target_token_index"]["\t"]] = 1

    result = ""

    while True:
        out, h, c = decoder.predict([target] + states, verbose=0)
        idx = np.argmax(out[0, -1, :])
        char = reverse[idx]

        if char == "\n":
            break

        result += char

        target = np.zeros((1, 1, data["num_decoder_tokens"]))
        target[0, 0, idx] = 1

        states = [h, c]

    return result


# -------- UI --------
st.title("Simple AutoCorrect")

encoder, decoder, data, reverse = load_all()

text = st.text_input("Enter text:")

if st.button("Correct"):
    if text:
        st.success(predict(text, encoder, decoder, data, reverse))
    else:
        st.warning("Enter something")


# -------- EXAMPLES --------
st.subheader("Examples")

st.table({
    "Input": ["speling", "eror", "machne", "lernng", "sentnce"],
    "Output": ["spelling", "error", "machine", "learning", "sentence"]
})