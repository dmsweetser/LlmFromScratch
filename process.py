# This script is a Python program that uses TensorFlow and Keras to create and train a neural network for text generation.
import numpy as np
import os
import tensorflow as tf

# Importing specific components from TensorFlow and Keras.
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# Importing text data from an external module named data.
from data import text_data_arr

# Create and configure the tokenizer with a specified vocabulary size.
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(text_data_arr)

# Convert all sequences, not just the first one.
sequences = tokenizer.texts_to_sequences(text_data_arr)

# Set the context length
context_length = 2048

# Create input and output sequences based on [/INST] tag.
input_sequences = []
output_sequences = []

for seq in sequences:
    for i in range(1, len(seq)):
        input_sequence = seq[:i+1]
        input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

        output_sequence = seq[i]
        
        # Append sequences to the input and output lists.
        input_sequences.append(input_padding)
        output_sequences.append(output_sequence)

# Convert lists to numpy arrays.
input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

# Determine the vocabulary size based on the unique words in the tokenizer.
vocab_size = len(tokenizer.word_index) + 1

# Define the sequence length for the generate_text function.
sequence_length = context_length

# Define the model architecture.
model = Sequential([
    Embedding(vocab_size, 32, input_length=context_length),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation="relu"),
    Dense(vocab_size, activation="softmax"),
])

# Compile the model with specified loss function, optimizer, and metrics.
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Checking if a pre-trained model exists. If yes, load it; otherwise, train a new model, save it, and print the respective messages.
existing_model_filename = "model.keras"

if os.path.exists(existing_model_filename):
    # Load existing model and continue training.
    model = tf.keras.models.load_model(existing_model_filename)
    print("Loaded existing model:", existing_model_filename)
else:
    # Train the model from scratch with shuffled input sequences.
    np.random.shuffle(input_sequences)
    epochs = 1
    batch_size = 32
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    print("Trained a new model")

    # Save the model.
    model.save(existing_model_filename)
    print("Saved the model as", existing_model_filename)

# Defining a function generate_text for generating text using the trained model.
# The function takes a seed text, model, tokenizer, sequence length, number of characters to generate, and an optional temperature parameter.
def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=1.0):
    generated_text = seed_text

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])
        token_list = pad_sequences(token_list, maxlen=context_length, padding="pre")
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

    return generated_text

seed_text = "2 + 2 = "

# Setting a seed text and generating text using the generate_text function.
# The generated text is printed to the console.
# The temperature parameter is used to control the level of randomness in the generated text.
generated_text = generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate=800, temperature=0.5)
print(generated_text)