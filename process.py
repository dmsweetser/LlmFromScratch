import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time
import json

training_data_file = "training_data.json"

def log_to_file(log_file_name, message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file_name, "a") as log_file:
        log_file.write(log_entry)

def generate_text(log_file_name, end_token, seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=0.87):
    start_time = time.time()

    generated_text = seed_text
    result = ""

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]

        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Calculate predicted token within the valid range of word_index
        valid_predicted_tokens = [index for index in range(1, len(tokenizer.word_index) + 1)]
        predicted_token = np.random.choice(valid_predicted_tokens, p=predicted_probs / np.sum(predicted_probs))

        # Find the corresponding word for the predicted token
        output_word = tokenizer.index_word.get(predicted_token, "")

        if output_word != "":
            result += output_word + " "
            if output_word == end_token:
                log_to_file(log_file_name, f"Detected end token '{end_token}'. Ending generation.")
                break

            generated_text += " " + output_word
            log_to_file(log_file_name, f"Current Result: {result}")

    end_time = time.time()
    time_taken = end_time - start_time
    log_to_file(log_file_name, f"Time taken for text generation: {time_taken} seconds")

    return result

def create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers):
    sequence_input = Input(shape=(context_length,), dtype='int32')
    embedded_sequence = Embedding(vocab_size, embedding_dim, input_length=context_length)(sequence_input)

    for _ in range(n_layers):
        lstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedded_sequence)
        attention_output = Attention()([lstm_output, lstm_output])
        conv_output = Conv1D(filters=hidden_dim, kernel_size=3, activation='relu')(attention_output)
        pooled_output = MaxPooling1D(pool_size=2)(conv_output)
        normalized_output = BatchNormalization()(pooled_output)
        gru_output = Bidirectional(GRU(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(normalized_output)
        lstm_attention_output = LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2)(gru_output)
        dense_output = Dense(hidden_dim, activation='relu')(lstm_attention_output)
        dropout_output = Dropout(0.2)(dense_output)
        output = Dense(vocab_size, activation='softmax')(dropout_output)

    model = Model(inputs=sequence_input, outputs=output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name):
    # Load existing training data from the JSON file if it exists
    existing_data = []
    if os.path.exists(training_data_file):
        with open(training_data_file, 'r') as json_file:
            existing_data = json.load(json_file)

    # Append new data to existing data
    all_text_data_arr = existing_data + text_data_arr

    # Update the tokenizer with the combined dataset
    tokenizer.fit_on_texts(all_text_data_arr)
    sequences = tokenizer.texts_to_sequences(all_text_data_arr)

    input_sequences = []
    output_sequences = []

    for seq in sequences:
        original_text = text_data_arr[sequences.index(seq)]
        parts = original_text.split(delimiter)
        question, answer = parts[0], parts[1]

        log_to_file(log_file_name, f"Original Text: {original_text}")
        log_to_file(log_file_name, f"Question: {question}")
        log_to_file(log_file_name, f"Answer: {answer}")

        question_sequence = tokenizer.texts_to_sequences([question])[0]
        answer_sequence = tokenizer.texts_to_sequences([answer])[0]

        log_to_file(log_file_name, f"Question Sequence: {question_sequence}")
        log_to_file(log_file_name, f"Answer Sequence: {answer_sequence}")

        for i in range(1, len(answer_sequence) + 1):
            input_sequence = question_sequence + answer_sequence[:i]
            input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

            output_sequence = answer_sequence[i - 1]

            input_sequences.append(input_padding)
            output_sequences.append(output_sequence)

    log_to_file(log_file_name, f"Input Sequences Shape: {np.array(input_sequences).shape}")
    log_to_file(log_file_name, f"Output Sequences Shape: {np.array(output_sequences).shape}")

    # Save training data to JSON file
    with open(training_data_file, 'w') as json_file:
        json.dump(text_data_arr, json_file)

    return np.array(input_sequences), np.array(output_sequences)

def train_model(model, input_sequences, output_sequences, epochs, batch_size, log_file_name):
    log_to_file(log_file_name, f"Input Sequences Shape: {input_sequences.shape}")
    log_to_file(log_file_name, f"Output Sequences Shape: {output_sequences.shape}")
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

def chat_loop(log_file_name, end_token, model, tokenizer, context_length, delimiter, num_chars_to_generate, epochs, batch_size):
    while True:

        # Initialize empty lists for input and output sequences
        input_sequences = []
        output_sequences = []

        user_question = input("You: ")
        log_to_file(log_file_name, f"User: {user_question}")

        if delimiter not in user_question:
            # Generate a response using the model
            generated_response = generate_text(log_file_name, end_token, user_question.lower(), model, tokenizer, context_length, num_chars_to_generate=context_length)
            log_to_file(log_file_name, f"Assistant: {generated_response}")

            # Ask if the answer is good or bad
            print(f"Assistant: {generated_response}")
            user_feedback = input("Is the answer good or bad? (Type 'good' or 'bad'): ")
            log_to_file(log_file_name, f"User Feedback: {user_feedback}")

            if user_feedback.lower() == 'bad':
                # Ask for the correct answer
                correct_answer = input("How should I have answered? Enter the correct response: ")
                log_to_file(log_file_name, f"Correct Answer: {correct_answer}")

                # Update the training data with the new question and answer
                text_data_arr = [f"{user_question} {delimiter} {correct_answer} {end_token}".lower()]
                input_sequences, output_sequences = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
                train_model(model, input_sequences, output_sequences, epochs, batch_size, log_file_name)
                log_to_file(log_file_name, "Trained existing model with new data")

                model.save("model.keras")
                tokenizer_config = tokenizer.to_json()
                with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
                    json_file.write(tokenizer_config)
                log_to_file(log_file_name, "Saved the trained model as model.keras")
        else:
            # Update the training data with the new question and answer
            log_to_file(log_file_name, f"Auto-training with new input: {user_question}")
            text_data_arr = [f"{user_question} {end_token}".lower()]
            input_sequences, output_sequences = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
            train_model(model, input_sequences, output_sequences, epochs, batch_size, log_file_name)
            log_to_file(log_file_name, "Trained a new model")

            model.save("model.keras")
            tokenizer_config = tokenizer.to_json()
            with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)
            log_to_file(log_file_name, "Saved the trained model as model.keras")

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    end_token = '[e]'
    delimiter = '[m]'

    text_data_arr = [
        f"What is your name? {delimiter} My name is Bob. {end_token}".lower(),
    ]

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_ticks = str(time.time()).replace(".", "_")
    log_file_name = f"chat_log_{current_date}_{current_ticks}.txt"

    context_length = 512
    embedding_dim = 64
    lstm_units = 128
    hidden_dim = 128
    vocab_size = 50000
    n_layers = 32

    epochs = 5
    batch_size = 32

    tokenizer = Tokenizer(lower=True, filters='')

    if os.path.exists("model.keras"):
        model = tf.keras.models.load_model("model.keras")
        tokenizer_config_path = "tokenizer_config.json"
        with open(tokenizer_config_path, "r", encoding="utf-8") as json_file:
            tokenizer_config_str = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config_str)
        log_to_file(log_file_name, f"Loaded existing model: model.keras")
    else:
        input_sequences, output_sequences = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
        model = create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers)
        train_model(model, input_sequences, output_sequences, epochs, batch_size, log_file_name)
        log_to_file(log_file_name, "Trained a new model")
        model.save("model.keras")
        tokenizer_config = tokenizer.to_json()
        with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)        
        log_to_file(log_file_name, "Saved the trained model as model.keras")

    chat_loop(log_file_name, end_token, model, tokenizer, context_length, delimiter, num_chars_to_generate=context_length, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()
