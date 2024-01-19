import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time
from utils import *

# Set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

end_token = "woot"

# Define the initial text data with question-answer pairs
text_data_arr = [
    "What is your name? garg My name is Bob. woot",
    "What is 2 + 2? garg 2 + 2 = 4. woot"
]

# Log file setup
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file_name = f"chat_log_{current_date}.txt"

# Declare the model as a global variable
model = None

# Tokenizes at the character level
tokenizer = Tokenizer(lower=True)

# # Architecture params - SMART AND SLOW
# context_length = 2048  # Updated context length
# dim = 1024 # 4096
# n_layers = 32 # 32
# hidden_dim = 128 # 14336
# vocab_size = 32000 # 32000

# Architecture params - DUMB AND FAST
context_length = 2048  # Updated context length
dim = 32 # 4096
n_layers = 4 # 32
hidden_dim = 8 # 14336
vocab_size = 32000 # 32000

# Training params
epochs = 10 # 100
batch_size = 32

if os.path.exists("model.keras"):
    model = tf.keras.models.load_model("model.keras")
    log_to_file(log_file_name, f"Loaded existing model: model.keras")
else:
    # Train the model from scratch with shuffled input sequences.
    tokenizer.fit_on_texts(text_data_arr)
    sequences = tokenizer.texts_to_sequences(text_data_arr)

    input_sequences = []
    output_sequences = []

    for seq in sequences:
        # Split the entry into question and answer using the original text
        original_text = text_data_arr[sequences.index(seq)]
        parts = original_text.split("garg")
        question, answer = parts[0], parts[1]

        # Tokenize the question and answer separately
        question_sequence = tokenizer.texts_to_sequences([question])[0]
        answer_sequence = tokenizer.texts_to_sequences([answer])[0]

        for i in range(1, len(answer_sequence) + 1):
            input_sequence = question_sequence + answer_sequence[:i]
            input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

            output_sequence = answer_sequence[i - 1]  # Use the i-th element of the answer_sequence

            input_sequences.append(input_padding)
            output_sequences.append(output_sequence)

    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    # Adjust embedding dimension and LSTM units
    embedding_dim = dim
    lstm_units = hidden_dim // 2  # Adjusted based on the hidden dimension

    # Input layer for the sequence
    sequence_input = Input(shape=(context_length,), dtype='int32')

    # Embedding layer
    embedded_sequence = Embedding(vocab_size, embedding_dim, input_length=context_length)(sequence_input)

    # Modify the model architecture based on the given parameters
    for _ in range(n_layers):
        # Bidirectional LSTM layer
        lstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedded_sequence)

        # Attention layer
        attention_output = Attention()([lstm_output, lstm_output])

        # Convolutional layer
        conv_output = Conv1D(filters=hidden_dim, kernel_size=3, activation='relu')(attention_output)

        # Max pooling layer
        pooled_output = MaxPooling1D(pool_size=2)(conv_output)

        # Batch normalization layer
        normalized_output = BatchNormalization()(pooled_output)

        # Bidirectional GRU layer
        gru_output = Bidirectional(GRU(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(normalized_output)

        # LSTM layer after attention
        lstm_attention_output = LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2)(gru_output)

        # Additional Dense layer
        dense_output = Dense(hidden_dim, activation='relu')(lstm_attention_output)

        # Dropout layer for regularization
        dropout_output = Dropout(0.2)(dense_output)

        # Output layer
        output = Dense(vocab_size, activation='softmax')(dropout_output)

        # Model
        model = Model(inputs=sequence_input, outputs=output)

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    log_to_file(log_file_name, "Trained a new model")

    # Save the trained model
    model.save("model.keras")
    log_to_file(log_file_name, "Saved the trained model as model.keras")

# Initial test requests
log_to_file(log_file_name, f"User: What is your name?")
generated_response = generate_text(log_file_name, end_token, "What is your name? garg", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
log_to_file(log_file_name, f"Assistant: {generated_response}")
log_to_file(log_file_name, f"User: What is 2 + 2?")
generated_response = generate_text(log_file_name, end_token, "What is 2 + 2? garg", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
log_to_file(log_file_name, f"Assistant: {generated_response}")

# Chat loop
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
