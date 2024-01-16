# This script is a Python program that uses TensorFlow and Keras to create and train a neural network for text generation.

# Importing necessary libraries:
# NumPy for numerical operations,
# os for operating system-related functions,
# TensorFlow for machine learning.
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

# Creating a tokenizer object and fitting it on the provided text data.
# The tokenizer is configured to operate at the character level and convert all text to lowercase.
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(text_data_arr)

# Converting the text data to sequences using the fitted tokenizer.
# The [0] indexing suggests that only the first sequence is considered.
sequences = tokenizer.texts_to_sequences(text_data_arr)[0]

# Set the context length
context_length = 2048

# Create input and output sequences based on [/INST] tag.
input_sequences = []
output_sequences = []

for text in text_data_arr:
    # Split the text after the [/INST] tag.
    parts = text.split("[/INST]")
    
    # Pad the input text at the end
    input_text = f"{parts[0]}[/INST]"
    input_padding = ' ' * (context_length - len(input_text))
    padded_input = input_text + input_padding[:max(0, len(input_padding))]

    # Pad the output text at the end
    output_text = parts[1]
    output_padding = ' ' * (context_length - len(output_text))
    padded_output = output_text + output_padding[:max(0, len(output_padding))]

   
    # Convert text to sequences using the tokenizer.
    input_sequence = tokenizer.texts_to_sequences([padded_input])[0]
    output_sequence = tokenizer.texts_to_sequences([padded_output])[0]
    
    # Append sequences to the input and output lists.
    input_sequences.append(input_sequence)
    output_sequences.append(output_sequence)

# Determine the vocabulary size based on the unique words in the tokenizer.
vocab_size = len(tokenizer.word_index) + 1

# Convert lists to numpy arrays.
input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)





# Defining the architecture of the neural network model.
# It consists of an Embedding layer, two LSTM layers, and a Dense layer with softmax activation.
model = Sequential([
    # Embedding layer that maps each word in the input sequence to a dense vector.
    Embedding(vocab_size, 32, input_length=context_length),
    # First LSTM layer with 128 units, returning a sequence of outputs for each time step.
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    # Second LSTM layer with 128 units, returning only the final output for the whole sequence.
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    # Dense layer with a softmax activation, outputting a probability distribution over the vocabulary.
    Dense(vocab_size, activation="softmax"),
])

# Compiling the model with specified loss function, optimizer, and metrics.
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Checking if a pre-trained model exists. If yes, load it; otherwise, train a new model, save it, and print the respective messages.
existing_model_filename = "model.keras"

if os.path.exists(existing_model_filename):
    # Load existing model and continue training.
    model = tf.keras.models.load_model(existing_model_filename)
    print("Loaded existing model:", existing_model_filename)
else:
    # Train the model from scratch.
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
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
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