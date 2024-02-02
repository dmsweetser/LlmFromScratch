import numpy as np
from bob_the_bot import *

def evaluate_model(chat_bot, tokenizer, context_length, end_token, questions, temperature, repetition_penalty):
    for question in questions:
        print(f"User: {question}")
        generated_response = chat_bot.generate_text(end_token, question, chat_bot.model, tokenizer, context_length, num_chars_to_generate=context_length, temperature=temperature, repetition_penalty=repetition_penalty)
        print(f"Assistant: {generated_response}")
        print()

# Define the range of values for temperature and repetition_penalty
temperature_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
repetition_penalty_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Load the trained model and tokenizer
config = {
    "context_length": 128,
    "embedding_dim": 16,
    "lstm_units": 128,
    "hidden_dim": 16,
    "epochs": 40,
    "batch_size": 64,
    "learning_rate": 0.01,
    "dropout": 0.2,
    "recurrent_dropout": 0.2
}

chat_bot = BobTheBot(config, True)  # Bypass chat loop
tokenizer = chat_bot.tokenizer
end_token = chat_bot.end_token
questions = ["What is your name?", "What is 2 + 2?"]

# Iterate through all possible options for temperature and repetition_penalty
for temperature in temperature_values:
    for repetition_penalty in repetition_penalty_values:
        print(f"Evaluating with Temperature={temperature}, Repetition Penalty={repetition_penalty}")
        evaluate_model(chat_bot, tokenizer, chat_bot.context_length, end_token, questions, temperature, repetition_penalty)
        print("------------------------------")
