import numpy as np
import tensorflow as tf

# Import necessary components from TensorFlow
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
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

# Define the model architecture
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

# Check if a pre-trained model exists
existing_model_filename = "model.h5"
if tf.keras.utils.model_to_dot(model).to_string() == tf.keras.utils.model_to_dot(tf.keras.models.load_model(existing_model_filename)).to_string():
    # Load existing model and continue training
    model = tf.keras.models.load_model(existing_model_filename)
    print("Loaded existing model:", existing_model_filename)
else:
    # Train the model from scratch
    epochs = 100
    batch_size = 32
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)
    print("Trained a new model")

    # Save the model
    model.save(existing_model_filename)
    print("Saved the model as", existing_model_filename)

# Evaluate the model and generate text with temperature scaling
def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=1.0):
    generated_text = seed_text

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature scaling
        predicted_probs = np.log(predicted_probs) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Sample the next token
        predicted_token = np.random.choice(len(predicted_probs), p=predicted_probs)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        generated_text += output_word

    return generated_text

seed_text = "John: How are you, Mike?"

# Generate text with temperature scaling (e.g., temperature=0.5 for less randomness)
generated_text = generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate=800, temperature=0.5)
print(generated_text)
