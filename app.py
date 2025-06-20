import streamlit as st
import tensorflow as tf
import pickle
import numpy as np

# Load model and tokenizer
model = tf.keras.models.load_model('word_predictor_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set max sequence length 
max_len = 222  
# Text generation function
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = token_list[-max_len:]  # truncate to max_len
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_len, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# Streamlit UI
st.title("Word Predictor")
seed_text = st.text_input("Enter some text:")

if seed_text:
    next_word = predict_next_word(seed_text)
    st.write(f"**Next predicted word:** `{next_word}`")
