# import sys
# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow.keras.preprocessing.text as keras_text
# import tensorflow.keras.preprocessing.sequence as keras_sequence
# sys.modules["keras.src.preprocessing.text"] = keras_text
# sys.modules["keras.src.preprocessing.sequence"] = keras_sequence
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import LSTM as OriginalLSTM
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# #Custom LSTM class to remove unsuppoerted 'time_major' argument
# class LSTM(OriginalLSTM):
#     def __init__(self, *args, **kwargs):
#         if 'time_major' in kwargs:
#             kwargs.pop('time_major')
#         super().__init__(*args, **kwargs)

# #Load the pretrained model using customLSTM 'time_major' argument
# model = load_model("next_word_lstm.keras", custom_objects={'LSTM':LSTM})


# #loading the tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)


# def predict_next_word(model, tokenizer, text, max_sequence_len):


#     # Tokenize the input text
#     token_list = tokenizer.texts_to_sequences([text])[0]
    
#     #pad the sequence to match the model's expected length
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

#     #predic the probability of next word
#     predicted = model.predict(token_list, verbose=0)

#     #Get the index with the highest probability
#     predicted_index = np.argmax(predicted, axis=1)[0]

#     #create a reverse mapping from index to word
#     reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

#     #Retrievee the corresponding word for the predicted dindex
#     next_word = reverse_word_index.get(predicted_index, "")
#     return next_word
    
# st.title("Next Word Prediction")
# input_text = st.text_input("Enter the sequence of words", "The sun is shining")
# if st.button("Predict: Next Word"):
#     max_sequence_len = model.input_shape[1]+1
#     next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len )
#     st.write("The predicted next word is: ", next_word)

import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


model = load_model('next_word_lstm.keras')

with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)


def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(sequence), axis=-1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""


st.title("Next Word Predictor")
st.write("Enter the phrase")


user_input = st.text_input("Enter your text:")


max_sequence_len = model.input_shape[1]

if st.button("Predict Next Word"):
    if user_input:
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        if next_word:
            st.success(f"Predicted next word: *{next_word}*")
        else:
            st.warning("Unable to predict, Try something different")
    else:
        st.error("Enter some text to predict the next word.")
