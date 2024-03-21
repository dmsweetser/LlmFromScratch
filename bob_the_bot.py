import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Embedding, Bidirectional, GRU, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Add
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time
import json
import string

class BobTheBot:
    def __init__(self, config, bypass_chat_loop, training_data_path, tokenizer_path, model_path):

        self.training_data_file = training_data_path
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        
        self.logs_folder = "logs"

        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_ticks = str(time.time()).replace(".", "_")
        self.log_file_name = os.path.join(self.logs_folder, f"chat_log_{current_date}_{current_ticks}.txt")

        # Create the logs folder if it doesn't exist
        os.makedirs(self.logs_folder, exist_ok=True)

        self.end_token = '[e]'
        self.delimiter = '[m]'

        self.bypass_chat_loop = bypass_chat_loop
        self.context_length = config["context_length"]
        self.embedding_dim = config["embedding_dim"]
        self.lstm_units = config["lstm_units"]
        self.hidden_dim = config["hidden_dim"]
        self.n_layers = config["n_layers"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.dropout = config["dropout"]
        self.recurrent_dropout = config["recurrent_dropout"]
        self.learning_rate = config["learning_rate"]
        self.temperature = config["temperature"]
        self.repetition_penalty = config["repetition_penalty"]
        
        self.last_generated_words = {}

        self.log_to_file(f"Current config:\n\n{json.dumps(config, indent=4)}")

        try:
            self.num_chars_to_generate = self.context_length
            self.tokenizer = Tokenizer(lower=True, filters='')
            self.model = self.load_or_train_model()
        except Exception as e:
            self.log_to_file(f"Exception encountered: {e}")

    def log_to_file(self, message):
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_entry = f"{timestamp} {message}\n"

        print(log_entry)

        with open(self.log_file_name, "a") as log_file:
            log_file.write(log_entry)

    def generate_text(self, seed_text):
        try:
            temperature = self.temperature
            repetition_penalty = self.repetition_penalty
            end_token = self.end_token
            model = self.model
            tokenizer = self.tokenizer
            sequence_length = self.context_length
            num_chars_to_generate = self.context_length
            self.log_to_file(f"Predicting using a temperature of {temperature} and a repetition_penalty of {repetition_penalty}")
            self.log_to_file(f"User: {seed_text}")

            # Preprocess seed text
            seed_text = seed_text.translate(str.maketrans('', '', string.punctuation))
            generated_text = seed_text.lower()
            split_generated_text = generated_text.split()
            for word in split_generated_text:
                self.last_generated_words[word] = True        

            result = ""

            # Generate text
            for _ in range(num_chars_to_generate):
                token_list = tokenizer.texts_to_sequences([generated_text])[0]
                token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

                predicted_probs = model.predict(token_list, verbose=0)[0]

                # Sample the predicted token using the probabilities
                predicted_token = np.random.choice(len(predicted_probs), p=predicted_probs)

                output_word = tokenizer.index_word.get(predicted_token, "")

                # Apply repetition penalty
                if output_word in self.last_generated_words:
                    predicted_probs[predicted_token] *= repetition_penalty

                if output_word != "":
                    result += output_word + " "
                    if output_word == end_token:
                        self.log_to_file(f"Detected end token '{end_token}'. Ending generation.")
                        break

                    generated_text += " " + output_word
                    self.last_generated_words[output_word] = True
                else:
                    self.log_to_file(f"Warning: Invalid token index: {predicted_token}")
                    self.log_to_file(f"Generated text: {generated_text}")
                    break

            self.log_to_file(f"Assistant: {result}")
            return result
        except Exception as e:
            self.log_to_file(f"Error occurred during text generation: {e}")
            return ""

    def create_model(self, context_length, vocab_size, embedding_dim, lstm_units, hidden_dim):
        
        # Calculate parameters for each layer
        embedding_params = vocab_size * embedding_dim
        lstm1_params = 4 * ((embedding_dim + lstm_units) * lstm_units)
        lstm2_params = 4 * ((lstm_units + lstm_units) * lstm_units)
        dense1_params = (lstm_units * hidden_dim) + hidden_dim
        dense2_params = (hidden_dim * vocab_size) + vocab_size
        output_dense_params = (hidden_dim * vocab_size) + vocab_size

        # Total parameters
        total_params = embedding_params + lstm1_params + lstm2_params + dense1_params + dense2_params + output_dense_params

        print("Total parameters:", total_params)
        
        inputs = Input(shape=(context_length,))
        pipeline = Embedding(vocab_size, embedding_dim)(inputs)        
        pipeline = LSTM(lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout, return_sequences=True)(pipeline)
        pipeline = LSTM(lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)(pipeline)
        pipeline = Dense(hidden_dim, activation='relu')(pipeline)
        pipeline = Dense(vocab_size)(pipeline)  # No activation here

        # Apply softmax activation here
        outputs = Dense(vocab_size, activation='softmax')(pipeline)

        # Define the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with sparse categorical crossentropy loss and Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return model

    def _get_positional_encoding(self, seq_length, d_model):
        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(seq_length)
        ])

        # Use sine and cosine functions to encode positional information
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # Apply sin to even indices
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # Apply cos to odd indices

        position_embedding = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        return position_embedding
    
    def preprocess_data(self, text_data_arr, tokenizer, context_length):
        existing_data = []
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'r') as json_file:
                existing_data = json.load(json_file)

        all_text_data_arr = existing_data + text_data_arr

        tokenizer.fit_on_texts(all_text_data_arr)
        sequences = tokenizer.texts_to_sequences(all_text_data_arr)

        vocab_size = len(tokenizer.word_index) + 1

        input_sequences = []
        output_sequences = []

        for sequence in sequences:
            original_text = all_text_data_arr[sequences.index(sequence)].lower()

            for i in range(1, len(sequence)):
                input_sequence = sequence[:i]
                input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

                output_sequence = sequence[i]

                input_sequences.append(input_padding)
                output_sequences.append(output_sequence)

        with open(self.training_data_file, 'w') as json_file:
            json.dump(all_text_data_arr, json_file, indent=4)

        return np.array(input_sequences), np.array(output_sequences), vocab_size

    def train_model(self, model, input_sequences, output_sequences, epochs, batch_size):
        model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

    def load_or_train_model(self):
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
            tokenizer_config_path = self.tokenizer_path
            with open(tokenizer_config_path, "r", encoding="utf-8") as json_file:
                tokenizer_config_str = json_file.read()
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config_str)
                self.tokenizer = tokenizer
                self.log_to_file("Loaded existing model and tokenizer")
        else:
            text_data_arr = []
            for filename in os.listdir("ingest"):
                
                try:
                    with open(os.path.join("ingest", filename), encoding="utf-8") as file:
                        text_data_arr.append(file.read())
                except Exception as e:
                    print(f"Error processing file '{filename}': {e}")
                    continue
            input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length)
            model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
            self.train_model(model, input_sequences, output_sequences, self.epochs, self.batch_size)
            self.log_to_file("Trained a new model")
            model.save(self.model_path)
            tokenizer_config = self.tokenizer.to_json()
            with open(self.tokenizer_path, "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)
            self.log_to_file("Saved the trained model as model.keras")
        return model

    def chat_loop(self):

        while True:
            # Initialize empty lists for input and output sequences
            input_sequences = []
            output_sequences = []

            user_question = input("You: ")

            if self.delimiter not in user_question:
                # Generate a response using the model
                generated_response = self.generate_text(self.end_token, user_question, self.model, self.tokenizer, self.context_length, num_chars_to_generate=self.context_length)

                # Ask if the answer is good or bad
                print(f"Assistant: {generated_response}")
                user_feedback = input("Is the answer good or bad? (Type 'good' or 'bad'): ")
                self.log_to_file(f"User Feedback: {user_feedback}")

                if user_feedback.lower() == 'bad':
                    # Ask for the correct answer
                    correct_answer = input("How should I have answered? Enter the correct response: ")
                    self.log_to_file(f"Correct Answer: {correct_answer}")

                    # Update the training data with the new question and answer
                    text_data_arr = []
                    line = f"{user_question} {correct_answer} {self.end_token}"
                    words = line.strip().split()
                    max_window_size = min(self.batch_size, len(words))
                    for i in range(1, max_window_size // 2 + 1):
                        if i < 4:
                            continue
                        for j in range(len(words) - i):
                            left = ' '.join(words[j:j+i])
                            right = ' '.join(words[j+i:j+i+1])
                            text_data_arr.append(f"{left} [m] {right}")
                    input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length)
                    self.model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
                    self.train_model(self.model, input_sequences, output_sequences, self.epochs, self.batch_size)
                    self.log_to_file("Trained existing model with new data")

                    self.model.save(self.model_path)
                    tokenizer_config = self.tokenizer.to_json()
                    with open(self.tokenizer_path, "w", encoding="utf-8") as json_file:
                        json_file.write(tokenizer_config)
                    self.log_to_file("Saved the trained model as model.keras")
            else:
                self.process_correction(user_question)


    def process_correction(self, user_question):
        # Update the training data with the new question and answer
        self.log_to_file(f"Auto-training with new input: {user_question}")
        text_data_arr = []
        line = f"{user_question} {self.end_token}"
        words = line.strip().split()
        max_window_size = min(self.batch_size, len(words))
        for i in range(1, max_window_size // 2 + 1):
            for j in range(len(words) - i):
                left = ' '.join(words[j:j+i])
                right = ' '.join(words[j+i:j+i+1])
                text_data_arr.append(f"{left} [m] {right}")
        input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length)
        self.model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
        self.train_model(self.model, input_sequences, output_sequences, self.epochs, self.batch_size)
        self.log_to_file("Retrained existing model")

        self.model.save(self.model_path)
        tokenizer_config = self.tokenizer.to_json()
        with open(self.tokenizer_path, "w", encoding="utf-8") as json_file:
            json_file.write(tokenizer_config)
        self.log_to_file(f"Saved the trained model as {self.model_path}")

    def main(self):

        for num in range(1,10,1):
            self.log_to_file(f"User: What is your name?")
            generated_response = self.generate_text(f"What is your name?")
            self.log_to_file(f"Assistant: {generated_response}")
            self.log_to_file(f"User: What is 2 + 2?")
            generated_response = self.generate_text(f"What is 2 + 2?")
            self.log_to_file(f"Assistant: {generated_response}")

        if self.bypass_chat_loop is False:
            self.chat_loop()
    
if __name__ == "__main__":

    config = {
        "context_length": 64,
        "n_layers": 1,
        "embedding_dim": 64,
        "lstm_units": 64, 
        "hidden_dim": 64,
        "epochs": 60,
        "batch_size": 64,
        "learning_rate": 0.01,
        "dropout": 0.2,
        "recurrent_dropout": 0.2,
        "temperature": 1.0,
        "repetition_penalty": 1.0
    }

    bob_the_bot = BobTheBot(config, False, "training_data.json", "tokenizer_config.json", "model.keras")
    bob_the_bot.main()
