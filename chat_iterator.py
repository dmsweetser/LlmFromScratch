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
            "embedding_dim": 16,
            "lstm_units": 70,
            "hidden_dim": 50, 
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.01,
            "dropout": 0.2,
            "recurrent_dropout": 0.2
        }

        chat_bot = BobTheBot(config, True, f"training_data{time.time()}.json", f"tokenizer_config{time.time()}.json", f"model{time.time()}.keras")        

        for run in range(1,10,1):
            response = chat_bot.generate_text(user_input, 10.0, 50.0)
            print(f"\n\n\n\n\n{response}\n\n\n\n\n")
                
    chat()

if __name__ == "__main__":
    main()
