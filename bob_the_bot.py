import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout, SimpleRNN
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time
import json

class BobTheBot:
    def __init__(self, config, bypass_chat_loop):

        self.training_data_file = "training_data.json"
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
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.dropout = config["dropout"]
        self.recurrent_dropout = config["recurrent_dropout"]
        self.learning_rate = config["learning_rate"]

        self.log_to_file(f"Current config:\n\n{config}")

        try:
            self.num_chars_to_generate = self.context_length
            self.tokenizer = Tokenizer(lower=True, filters='')
            self.model = self.load_or_train_model()
        except Exception as e:
            self.log_to_file(f"Exception encountered for variation: {e}")

    def log_to_file(self, message):
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_entry = f"{timestamp} {message}\n"

        with open(self.log_file_name, "a") as log_file:
            log_file.write(log_entry)

    def generate_text(self, end_token, seed_text, model, tokenizer, sequence_length, num_chars_to_generate, temperature=0.87, repetition_penalty=1.01):

        self.last_generated_words = {}
        result = ""

        generated_text = seed_text.lower()

        split_generated_text = generated_text.split()
        for word in split_generated_text:
            self.last_generated_words[word] = True        

        for _ in range(num_chars_to_generate):
            token_list = tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences([token_list], maxlen=sequence_length, padding="pre")

            predicted_probs = model.predict(token_list, verbose=0)[0]

            predicted_probs = np.log(predicted_probs) / temperature
            exp_preds = np.exp(predicted_probs)
            predicted_probs = exp_preds / np.sum(exp_preds)

            valid_predicted_tokens = [index for index in range(1, len(tokenizer.word_index) + 1) if index not in self.last_generated_words]

            # Apply repetition penalty
            if len(self.last_generated_words) > 0:
                last_generated_word = list(self.last_generated_words.keys())[-1]
                last_generated_word_index = tokenizer.word_index[last_generated_word]
                last_generated_word_probability = predicted_probs[last_generated_word_index]

                if last_generated_word_probability > 0:
                    repetition_penalty = (1 - last_generated_word_probability) ** repetition_penalty
                    predicted_probs *= repetition_penalty

            # Addresses a casting issue later on
            predicted_probs = predicted_probs.astype(np.float64)

            # Ensure predicted_probs sum to 1
            predicted_probs /= np.sum(predicted_probs)

            # Calculate predicted token within the valid range of word_index
            predicted_token = np.argmax(np.random.multinomial(1, predicted_probs, 1)[0])

            # Find the corresponding word for the predicted token
            output_word = tokenizer.index_word.get(predicted_token, "")

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

        return result

    def create_model(self, context_length, vocab_size, embedding_dim, lstm_units, hidden_dim):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=context_length))
        model.add(LSTM(lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        model.add(Dense(hidden_dim, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(vocab_size, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def preprocess_data(self, text_data_arr, tokenizer, context_length, delimiter):
        # Load existing training data from the JSON file if it exists
        existing_data = []
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'r') as json_file:
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
            original_text = all_text_data_arr[sequences.index(seq)].lower()
            parts = original_text.split(delimiter)
            question, answer = parts[0], parts[1]

            self.log_to_file(f"PREPROCESSING - Original Text: {original_text}")
            self.log_to_file(f"PREPROCESSING - Question: {question}")
            self.log_to_file(f"PREPROCESSING - Answer: {answer}")

            question_sequence = tokenizer.texts_to_sequences([question])[0]
            answer_sequence = tokenizer.texts_to_sequences([answer])[0]

            self.log_to_file(f"PREPROCESSING - Question Sequence: {question_sequence}")
            self.log_to_file(f"PREPROCESSING - Answer Sequence: {answer_sequence}")

            for i in range(1, len(answer_sequence) + 1):
                input_sequence = question_sequence + answer_sequence[:i]
                input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

                output_sequence = answer_sequence[i - 1]

                input_sequences.append(input_padding)
                output_sequences.append(output_sequence)

        self.log_to_file(f"PREPROCESSING - Input Sequences Shape: {np.array(input_sequences).shape}")
        self.log_to_file(f"PREPROCESSING - Output Sequences Shape: {np.array(output_sequences).shape}")

        # Save training data to JSON file
        with open(self.training_data_file, 'w') as json_file:
            json.dump(all_text_data_arr, json_file)

        return np.array(input_sequences), np.array(output_sequences), vocab_size

    def train_model(self, model, input_sequences, output_sequences, epochs, batch_size):
        self.log_to_file(f"Input Sequences Shape: {input_sequences.shape}")
        self.log_to_file(f"Output Sequences Shape: {output_sequences.shape}")
        model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

    def load_or_train_model(self):
        if os.path.exists("model.keras"):
            model = tf.keras.models.load_model("model.keras")
            tokenizer_config_path = "tokenizer_config.json"
            with open(tokenizer_config_path, "r", encoding="utf-8") as json_file:
                tokenizer_config_str = json_file.read()
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config_str)
                self.tokenizer = tokenizer
                self.log_to_file("Loaded existing model and tokenizer")
        else:
            text_data_arr = [
                f"What is your name? {self.delimiter} My name is Bob. {self.end_token}",
                f"What is 2 + 2? {self.delimiter} 2 + 2 = 4. {self.end_token}"
                ]
            input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length, self.delimiter)
            model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
            self.train_model(model, input_sequences, output_sequences, self.epochs, self.batch_size)
            self.log_to_file("Trained a new model")
            model.save("model.keras")
            tokenizer_config = self.tokenizer.to_json()
            with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)
            self.log_to_file("Saved the trained model as model.keras")
        return model

    def chat_loop(self):

        while True:
            # Initialize empty lists for input and output sequences
            input_sequences = []
            output_sequences = []

            user_question = input("You: ")
            self.log_to_file(f"User: {user_question}")

            if self.delimiter not in user_question:
                # Generate a response using the model
                generated_response = self.generate_text(self.end_token, user_question, self.model, self.tokenizer, self.context_length, num_chars_to_generate=self.context_length)
                self.log_to_file(f"Assistant: {generated_response}")

                # Ask if the answer is good or bad
                print(f"Assistant: {generated_response}")
                user_feedback = input("Is the answer good or bad? (Type 'good' or 'bad'): ")
                self.log_to_file(f"User Feedback: {user_feedback}")

                if user_feedback.lower() == 'bad':
                    # Ask for the correct answer
                    correct_answer = input("How should I have answered? Enter the correct response: ")
                    self.log_to_file(f"Correct Answer: {correct_answer}")

                    # Update the training data with the new question and answer
                    text_data_arr = [f"{user_question} {self.delimiter} {correct_answer} {self.end_token}"]
                    input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length, self.delimiter)
                    self.model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
                    self.train_model(self.model, input_sequences, output_sequences, self.epochs, self.batch_size)
                    self.log_to_file("Trained existing model with new data")

                    self.model.save("model.keras")
                    tokenizer_config = self.tokenizer.to_json()
                    with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
                        json_file.write(tokenizer_config)
                    self.log_to_file("Saved the trained model as model.keras")
            else:
                # Update the training data with the new question and answer
                self.log_to_file(f"Auto-training with new input: {user_question}")
                text_data_arr = [f"{user_question} {self.end_token}"]
                input_sequences, output_sequences, vocab_size = self.preprocess_data(text_data_arr, self.tokenizer, self.context_length, self.delimiter)
                self.model = self.create_model(self.context_length, vocab_size, self.embedding_dim, self.lstm_units, self.hidden_dim)
                self.train_model(self.model, input_sequences, output_sequences, self.epochs, self.batch_size)
                self.log_to_file("Retrained existing model")

                self.model.save("model.keras")
                tokenizer_config = self.tokenizer.to_json()
                with open("tokenizer_config.json", "w", encoding="utf-8") as json_file:
                    json_file.write(tokenizer_config)
                self.log_to_file("Saved the trained model as model.keras")

    def main(self):

        # self.log_to_file(f"User: What is your name?")
        # generated_response = self.generate_text(self.end_token, f"What is your name?", self.model, self.tokenizer, self.context_length, num_chars_to_generate=self.context_length)
        # self.log_to_file(f"Assistant: {generated_response}")
        # self.log_to_file(f"User: What is 2 + 2?")
        # generated_response = self.generate_text(self.end_token, f"What is your 2 + 2?", self.model, self.tokenizer, self.context_length, num_chars_to_generate=self.context_length)
        # self.log_to_file(f"Assistant: {generated_response}")

        if self.bypass_chat_loop is False:
            self.chat_loop()

if __name__ == "__main__":

    config = {
      "context_length": 128,
      "embedding_dim": 16,
      "lstm_units": 128,
      "hidden_dim": 16,
      "epochs": 100,
      "batch_size": 64,
      "learning_rate": 0.01,
      "dropout": 0.2,
      "recurrent_dropout": 0.2
    }

    bob_the_bot = BobTheBot(config, False)
    bob_the_bot.main()
