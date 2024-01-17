import numpy as np
import os
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import datetime
import time

# Set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Attention = tf.keras.layers.Attention

# Define the initial text data
text_data_arr = ["[INST]What is your name?[/INST]My name is Bob.<s>"]

context_length = 512

# Log file setup
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file_name = f"chat_log_{current_date}.txt"

# Declare the model as a global variable
model = None

tokenizer = Tokenizer(char_level=True, lower=True)  # Initialize tokenizer

def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file_name, "a") as log_file:
        log_file.write(log_entry)

def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=1.0):
    start_time = time.time()

    generated_text = [f"[INST]{seed_text}[/INST]"]


    for _ in range(num_chars_to_generate):

        # Check if "<s>" is in the generated text
        if "<s>" in generated_text:
            break

        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Apply temperature scaling.
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Sample the next token.
        predicted_token = np.random.choice(len(predicted_probs), p=predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        generated_text += output_word
        if output_word != "":
            print(f"Current Result: {generated_text}")
            log_to_file(f"Current Result: {generated_text}")

    end_time = time.time()
    time_taken = end_time - start_time
    log_to_file(f"Time taken for text generation: {time_taken} seconds")

    return generated_text

if os.path.exists("model.keras"):
    model = tf.keras.models.load_model("model.keras")
    log_to_file(f"Loaded existing model: model.keras")
else:
    # Train the model from scratch with shuffled input sequences.
    tokenizer.fit_on_texts(text_data_arr)
    sequences = tokenizer.texts_to_sequences(text_data_arr)

    input_sequences = []
    output_sequences = []

    for seq in sequences:
        for i in range(1, len(seq)):
            input_sequence = seq[:i + 1]
            input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

            output_sequence = seq[i]

            input_sequences.append(input_padding)
            output_sequences.append(output_sequence)

    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential([
        Embedding(vocab_size, 32, input_length=context_length),
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(vocab_size, activation="softmax"),
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    epochs = 10
    batch_size = 32
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    log_to_file("Trained a new model")

    # Save the trained model
    model.save("model.keras")
    log_to_file("Saved the trained model as model.keras")

# Chat loop
while True:
    user_question = input("You: ")
    log_to_file(f"User: {user_question}")

    # Generate a response using the model
    generated_response = generate_text(user_question, model, tokenizer, context_length, num_chars_to_generate=800, temperature=0.5)
    print("Assistant:", generated_response)
    log_to_file(f"Assistant: {generated_response}")

    # Ask if the answer is good or bad
    user_feedback = input("Is the answer good or bad? (Type 'good' or 'bad'): ")
    log_to_file(f"User Feedback: {user_feedback}")

    if user_feedback.lower() == 'bad':
        # Ask for the correct answer
        correct_answer = input("How should I have answered? Enter the correct response: ")
        log_to_file(f"Correct Answer: {correct_answer}")

        # Update the training data with the new question and answer
        new_data = [f"[INST]{seed_text}[/INST]{correct_answer}<s>"]
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
        log_to_file("Model retrained with the updated data")

    # Optionally, add an exit condition for the chat loop
    exit_chat_loop = input("Do you want to exit the chat loop? (Type 'yes' to exit): ")
    log_to_file(f"Exit Chat Loop: {exit_chat_loop}")
    if exit_chat_loop.lower() == 'yes':
        break
