import os
import subprocess
import json
import sys
from bob_the_bot import BobTheBot
import time
import tensorflow as tf

def main():

    def chat():
        user_input = "What is your name?"

        config = {
            "context_length": 256,
            "n_layers": 1,
            "embedding_dim": 128,
            "lstm_units": 128, 
            "hidden_dim": 100000,
            "epochs": 60,
            "batch_size": 64,
            "learning_rate": 0.01,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        }

        chat_bot = BobTheBot(
            config, 
            True, 
            f"training_data_constitution.json", 
            f"tokenizer_config{time.time()}.json", 
            f"model{time.time()}.keras", 
            "ingest_2"
            )        

        for run in range(1,10,1):
            response = chat_bot.generate_text("What is your name?")
            print(f"\n\n\n\n\n{response}\n\n\n\n\n")
            
        for run in range(1,10,1):
            response = chat_bot.generate_text("What is 2 + 4?")
            print(f"\n\n\n\n\n{response}\n\n\n\n\n")
                
    chat()

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()
