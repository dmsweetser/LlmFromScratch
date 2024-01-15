import numpy as np
import tensorflow as tf

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
SimpleRNN = tf.keras.layers.SimpleRNN
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout

# Load your text data
# Here I'm simply loading a relative file which contains the array of data (data.py)
from data import text_data_arr

# Tokenize the text
tokenizer = Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(text_data_arr)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(text_data_arr)[0]

# Prepare input and target sequences
input_sequences = []
output_sequences = []

sequence_length = 100
for i in range(len(sequences) - sequence_length):
    input_sequences.append(sequences[i:i + sequence_length])
    output_sequences.append(sequences[i + sequence_length])

input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

vocab_size = len(tokenizer.word_index) + 1

# Define the model architecture:
model = Sequential([
    # Embedding layer that maps each word in the input sequence to a dense vector
    Embedding(vocab_size, 32, input_length=sequence_length),
    # First LSTM layer with 128 units, returning a sequence of outputs for each time step
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    # Second LSTM layer with 128 units, returning only the final output for the whole sequence
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    # Dense layer with a softmax activation, outputting a probability distribution over the vocabulary
    Dense(vocab_size, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Train the model
epochs = 100  # Increase the number of epochs to give the model more time to learn
batch_size = 32
model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

model.save()

# Evaluate the model and generate text:
def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate):
    generated_text = seed_text

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_token = np.argmax(predicted_probs, axis=-1)[0]  # Get the index of the predicted token

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        generated_text += output_word

    return generated_text

seed_text = "John: How are you, Mike?"

generated_text = generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate=800)
print(generated_text)