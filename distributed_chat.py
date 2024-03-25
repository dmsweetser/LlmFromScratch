import os
import subprocess
import json
import sys
from bob_the_bot import BobTheBot
import time
import tensorflow as tf
import numpy as np

def main():

    def chat():

        config = {
            "context_length": 64,
            "embedding_dim": 64,
            "lstm_units": 64, 
            "hidden_dim": 64,
            "epochs": 40,
            "batch_size": 64,
            "learning_rate": 0.015,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "temperature": 1.0,
            "repetition_penalty": 1.0
        }

        # chat_bot_1 = BobTheBot(
        #     config, 
        #     True, 
        #     f"training_data{time.time()}_1.json", 
        #     f"tokenizer_config{time.time()}_1.json", 
        #     f"model{time.time()}_1.keras",
        #     "ingest_1"
        #     )        
        
        chat_bot_2 = BobTheBot(
            config, 
            True, 
            f"training_data{time.time()}_2.json", 
            f"tokenizer_config{time.time()}_2.json", 
            f"model{time.time()}_2.keras",
            "ingest_2"
            )   
        
        chat_bot_3 = BobTheBot(
            config, 
            True, 
            f"training_data{time.time()}_3.json", 
            f"tokenizer_config{time.time()}_3.json", 
            f"model{time.time()}_3.keras",
            "ingest_3"
            )   
        
        seed_text = "What is 2 + 2?"
        result = ""
        while True:
            #result_1, result_1_prob = chat_bot_1.get_next_token(seed_text)
            result_2, result_2_prob = chat_bot_2.get_next_token(seed_text)
            result_3, result_3_prob = chat_bot_3.get_next_token(seed_text)

            # Collect results and probabilities
            results = [result_2, result_3]
            probabilities = [result_2_prob, result_3_prob]
            
            print(results)
            print(probabilities)

            # Find the index of the maximum probability
            max_prob_index = np.argmax(probabilities)

            # Append the corresponding value to the seed text with a space
            seed_text += " " + results[max_prob_index]
            result += " " + results[max_prob_index]
            
            print(result)
            
            if "[e]" in result:
                break
            
    chat()

if __name__ == "__main__":
    main()
