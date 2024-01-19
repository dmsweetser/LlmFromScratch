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

def generate_text(log_file_name, end_token, seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=1.0, repetition_penalty=1.01):
    start_time = time.time()

    generated_text = seed_text
    result = ""

    tokenizer.fit_on_texts([generated_text])

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]

        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        penalty_adjustment = np.ones_like(predicted_probs)
        for word, index in tokenizer.word_index.items():
            if word in generated_text:
                penalty_adjustment[index - 1] = repetition_penalty

        predicted_probs = predicted_probs * penalty_adjustment

        predicted_token = np.random.choice(len(predicted_probs), p=predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        print(output_word)

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

def chat_loop(log_file_name, end_token, model, tokenizer, context_length, num_chars_to_generate, epochs, batch_size):
    input_sequences = np.array([])
    output_sequences = np.array([])

    while True:
        user_question = input("You: ")
        log_to_file(log_file_name, f"User: {user_question}")

        # Generate a response using the model
        generated_response = generate_text(log_file_name, end_token, user_question, model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
        print("Assistant:", generated_response)
        log_to_file(log_file_name, f"Assistant: {generated_response}")

        # Ask if the answer is good or bad
        user_feedback = input("Is the answer good or bad? (Type 'good' or 'bad'): ")
        log_to_file(log_file_name, f"User Feedback: {user_feedback}")

        if user_feedback.lower() == 'bad':
            # Ask for the correct answer
            correct_answer = input("How should I have answered? Enter the correct response: ")
            log_to_file(log_file_name, f"Correct Answer: {correct_answer}")

            # Update the training data with the new question and answer
            new_data = [f"{user_question} garg {correct_answer} woot"]
            new_sequences = tokenizer.texts_to_sequences(new_data)

            for seq in new_sequences:
                for i in range(1, len(seq)):
                    input_sequence = seq[:i + 1]
                    output_sequence = seq[i]

                    # Ensure the input sequence is padded to the correct length
                    input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

                    input_sequences = np.append(input_sequences, [input_padding], axis=0)
                    output_sequences = np.append(output_sequences, [output_sequence])

            # Retrain the model with the updated data
            model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
            model.save("model.keras")
            log_to_file(log_file_name, "Model retrained with the updated data")

        # Optionally, add an exit condition for the chat loop
        exit_chat_loop = input("Do you want to exit the chat loop? (Type 'yes' to exit): ")
        log_to_file(log_file_name, f"Exit Chat Loop: {exit_chat_loop}")
        if exit_chat_loop.lower() == 'yes':
            break