import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time


def log_to_file(log_file_name, message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file_name, "a") as log_file:
        log_file.write(log_entry)

def generate_text(log_file_name, end_token, seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=0.87, repetition_penalty=2.0):
    start_time = time.time()

    generated_text = seed_text
    result = ""

    for _ in range(num_chars_to_generate):
        
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Apply temperature scaling.
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Apply repetition penalty
        penalty_adjustment = np.ones_like(predicted_probs)
        for word, index in tokenizer.word_index.items():
            if word in generated_text:
                penalty_adjustment[index - 1] = repetition_penalty

        predicted_probs = predicted_probs * penalty_adjustment

        # Normalize probabilities to ensure they sum to 1
        predicted_probs /= predicted_probs.sum()

        # Sample the next token.
        predicted_token = np.random.choice(len(predicted_probs), p=predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        result += output_word + " "
        if output_word == end_token:
            print(f"Detected end token '{end_token}'. Ending generation.")
            break

        if output_word != "":
            generated_text += " " + output_word
            print(f"Current Result: {result}")

    end_time = time.time()
    time_taken = end_time - start_time
    log_to_file(log_file_name, f"Time taken for text generation: {time_taken} seconds")

    return result