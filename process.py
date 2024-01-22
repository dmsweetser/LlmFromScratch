import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time
import json

training_data_file = "training_data.json"
logs_folder = "logs"

def log_to_file(log_file_path, message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file_path, "a") as log_file:
        log_file.write(log_entry)

def generate_text(log_file_path, end_token, seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=0.87):
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
        print(f"Valid Predicted Tokens: {valid_predicted_tokens}")
        print(f"Predicted Probabilities: {predicted_probs}")

        # Ensure predicted_probs sum to 1
        predicted_probs /= np.sum(predicted_probs)

        # Calculate predicted token without using np.random.choice
        predicted_token = np.argmax(np.random.multinomial(1, predicted_probs, 1))

        print(f"Predicted Token: {predicted_token}")

        # Find the corresponding word for the predicted token
        output_word = tokenizer.index_word.get(predicted_token, "")

        print(f"Output Word: {output_word}")

        if output_word != "":
            result += output_word + " "
            if output_word == end_token:
                log_to_file(log_file_path, f"Detected end token '{end_token}'. Ending generation.")
                break

            generated_text += " " + output_word
            log_to_file(log_file_path, f"Current Result: {result}")

    end_time = time.time()
    time_taken = end_time - start_time
    log_to_file(log_file_path, f"Time taken for text generation: {time_taken} seconds")

    return result

'''
MODEL DESCRIPTION

Imagine your model is like a storyteller, and it's really good at understanding and creating stories. The stories are made up of words, and each word has its own special meaning. The model's job is to learn how to tell stories by understanding the patterns and relationships between these words.

Now, think of the Embedding layer as a magical dictionary that helps the storyteller understand the meaning of each word. When the storyteller reads a word, it looks up the word in this magical dictionary and gets a special code that represents the word's meaning. This code is like a secret language that the storyteller and the dictionary use to communicate.

In our story, the storyteller (model) wants to create really interesting and detailed stories. So, it has a special trick called Bidirectional LSTM. This is like having a friend who reads the story from the beginning to the end and another friend who reads it from the end to the beginning. They both share their understanding, and it helps the storyteller catch all the important details and connections in the story.

Then, there's an Attention mechanism, which is like a spotlight that the storyteller uses to focus on the most exciting parts of the story. It helps the storyteller pay extra attention to important details.

After that, there's a Convolutional layer, which is like a chef adding some spice to the story. It enhances certain aspects of the story to make it more flavorful and interesting.

The BatchNormalization is like having a helper who ensures that everything stays in order and doesn't get too messy. It helps keep the story well-balanced.

Now, the GRU (Gated Recurrent Unit) is like having another set of friends who remember bits of the story and share them with the storyteller. They work together to make sure no part of the story is forgotten.

Finally, the storyteller puts everything together and tells the story. The Dense layer is like the storyteller organizing all the information and presenting it in a way that makes sense.

So, in a nutshell, the model is like a storyteller with magical dictionaries, friends who read in both directions, spotlights, chefs, helpers, and memory-keeping friends. All of them work together to create the best and most exciting stories!
'''

def create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers):
    sequence_input = Input(shape=(context_length,), dtype='int32')
    embedded_sequence = Embedding(vocab_size, embedding_dim, input_length=context_length)(sequence_input)

    lstm_output = embedded_sequence
    for _ in range(n_layers):
        lstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm_output)

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

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for the padding token

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

    return np.array(input_sequences), np.array(output_sequences), vocab_size

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
                input_sequences, output_sequences, vocab_size = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
                model = create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers)
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
            input_sequences, output_sequences, vocab_size = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
            model = create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers)
            train_model(model, input_sequences, output_sequences, epochs, batch_size, log_file_name)
            log_to_file(log_file_name, "Retrained existing model")

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
    log_file_name = f"{logs_folder}/chat_log_{current_date}_{current_ticks}.txt"

    # Create the logs folder if it doesn't exist
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    context_length = 512
    embedding_dim = 64
    lstm_units = 128
    hidden_dim = 128
    n_layers = 2

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
        input_sequences, output_sequences, vocab_size = preprocess_data(text_data_arr, tokenizer, context_length, delimiter, log_file_name)
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
