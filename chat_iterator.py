import os
import subprocess
import json
import sys
from bob_the_bot import BobTheBot
import time

def main():

    def chat():
        user_input = "What is your name?"

        config = {
            "context_length": 64,
            "n_layers": 512,
            "embedding_dim": 128,
            "lstm_units": 512, 
            "hidden_dim": 512,
            "epochs": 10,
            "batch_size": 64,
            "learning_rate": 0.01,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        }

        chat_bot = BobTheBot(config, True, f"training_data{time.time()}.json", f"tokenizer_config{time.time()}.json", f"model{time.time()}.keras")        

        for run in range(1,10,1):
            response = chat_bot.generate_text(user_input)
            print(f"\n\n\n\n\n{response}\n\n\n\n\n")
                
    chat()

if __name__ == "__main__":
    main()
