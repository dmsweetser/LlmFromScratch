import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Bidirectional, GRU, Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime
import time

def create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers):
    sequence_input = Input(shape=(context_length,), dtype='int32')
    embedded_sequence = Embedding(vocab_size, embedding_dim, input_length=context_length)(sequence_input)

    for _ in range(n_layers):
        lstm_output = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedded_sequence)
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

def preprocess_data(text_data_arr, tokenizer, context_length):
    tokenizer.fit_on_texts(text_data_arr)
    sequences = tokenizer.texts_to_sequences(text_data_arr)

    input_sequences = []
    output_sequences = []

    for seq in sequences:
        original_text = text_data_arr[sequences.index(seq)]
        parts = original_text.split("garg")
        question, answer = parts[0], parts[1]

        question_sequence = tokenizer.texts_to_sequences([question])[0]
        answer_sequence = tokenizer.texts_to_sequences([answer])[0]

        for i in range(1, len(answer_sequence) + 1):
            input_sequence = question_sequence + answer_sequence[:i]
            input_padding = pad_sequences([input_sequence], maxlen=context_length, padding="pre")[0]

            output_sequence = answer_sequence[i - 1]

            input_sequences.append(input_padding)
            output_sequences.append(output_sequence)

    return np.array(input_sequences), np.array(output_sequences)

def train_model(model, input_sequences, output_sequences, epochs, batch_size):
    model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    end_token = "woot"

    text_data_arr = [
        "What is your name? garg My name is Bob. woot",
        "What is 2 + 2? garg 2 + 2 = 4. woot"
    ]

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"chat_log_{current_date}.txt"

    # SMART AND SLOW
    # context_length = 2048
    # embedding_dim = 32
    # lstm_units = 128
    # hidden_dim = 128
    # vocab_size = 32000
    # n_layers = 32

    # DUMB AND FAST
    context_length = 2048
    embedding_dim = 32
    lstm_units = 8
    hidden_dim = 8
    vocab_size = 32000
    n_layers = 4

    epochs = 10
    batch_size = 32

    tokenizer = Tokenizer(lower=True)

    if os.path.exists("model.keras"):
        model = tf.keras.models.load_model("model.keras")
        log_to_file(log_file_name, f"Loaded existing model: model.keras")
    else:
        input_sequences, output_sequences = preprocess_data(text_data_arr, tokenizer, context_length)
        model = create_model(context_length, vocab_size, embedding_dim, lstm_units, hidden_dim, n_layers)
        train_model(model, input_sequences, output_sequences, epochs, batch_size)
        log_to_file(log_file_name, "Trained a new model")

        model.save("model.keras")
        log_to_file(log_file_name, "Saved the trained model as model.keras")

    # Initial test requests
    log_to_file(log_file_name, f"User: What is your name?")
    generated_response = generate_text(log_file_name, end_token, "What is your name? garg", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
    log_to_file(log_file_name, f"Assistant: {generated_response}")
    log_to_file(log_file_name, f"User: What is 2 + 2?")
    generated_response = generate_text(log_file_name, end_token, "What is 2 + 2? garg", model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=1.0)
    log_to_file(log_file_name, f"Assistant: {generated_response}")

    chat_loop(log_file_name, end_token, model, tokenizer, context_length, num_chars_to_generate=context_length, epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    main()
