import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time

# Set the environment variable TF_ENABLE_ONEDNN_OPTS to 0
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

end_token = "[END]"

# Define the initial text data with question-answer pairs
text_data_arr = [
    "What is your name? [A] My name is Bob. [END]",
    "What is 2 + 2? [A] 2 + 2 = 4. [END]"
]

context_length = 2048  # Updated context length

# Log file setup
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file_name = f"chat_log_{current_date}.txt"

# Declare the model as a global variable
model = None

# Tokenizes at the word level
tokenizer = Tokenizer(lower=True)  # Initialize tokenizer

def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}\n"

    with open(log_file_name, "a") as log_file:
        log_file.write(log_entry)

def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=0.87, repetition_penalty=2.0):
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
    log_to_file(f"Time taken for text generation: {time_taken} seconds")

    return result

# Modify the architecture based on the given parameters
dim = 1024 # 4096
n_layers = 32 # 32
hidden_dim = 128 # 14336
vocab_size = 32000 # 32000

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
        # Split the entry into question and answer using the original text
        original_text = text_data_arr[sequences.index(seq)]
        parts = original_text.split("[A]")
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

    epochs = 100
    batch_size = 32
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    log_to_file("Trained a new model")

    # Save the trained model
    model.save("model.keras")
    log_to_file("Saved the trained model as model.keras")

# Initial test requests
log_to_file(f"User: What is your name?")
generated_response = generate_text("What is your name? [A]", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
log_to_file(f"Assistant: {generated_response}")
log_to_file(f"User: What is 2 + 2?")
generated_response = generate_text("What is 2 + 2? [A]", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
log_to_file(f"Assistant: {generated_response}")

# Chat loop
while True:
    user_question = input("You: ")
    log_to_file(f"User: {user_question}")

    # Generate a response using the model
    generated_response = generate_text(user_question, model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
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
        new_data = [f"{user_question} [A] {correct_answer} [END]"]
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
