import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load the model
model = load_model('lstm_model.h5') 

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    return tokenizer.index_word[predicted_word_index]   
##streamlit app
st.title("Next Word Prediction")
input_text = st.text_input("Enter text:", "Speake of it. Stay, and speake. Stop it  ")
if st.button("Predict"):
    max_sequence_len = 10  # Adjust this based on your model's training
    predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted next word for '{input_text}': {predicted_word}")

