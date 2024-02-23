import os
import subprocess
import json
import sys
from bob_the_bot import BobTheBot

def main():


    config = {
        "context_length": 64,
        "n_layers": 30,
        "embedding_dim": 16,
        "lstm_units": 70,
        "hidden_dim": 50,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.01,
        "dropout": 0.2,
        "recurrent_dropout": 0.2
    }

    chat_bot = BobTheBot(config, True, "training_data_1.json", "tokenizer_config_1.json", "model_1.keras")

    def chat():
        while True:
            user_input = input("\nHow can I help?\n")
            temp = 0.1
            repetition_penalty = 0.2
            while temp < 20.0:
                while repetition_penalty < 20.0:
                    response = chat_bot.generate_text(user_input, temp, repetition_penalty)
                    if "[e]" not in response:
                        response = chat_bot.generate_text(f"{user_input} {response}", temp / 2, repetition_penalty * 2)
                    if "name is bob. [e]" in response:
                        print(f"Temp: {temp}")
                        print(f"Repetition Penalty: {repetition_penalty}")
                        print(f"\n\n\n{response}\n\n\n")
                        break

                    # feedback = input("\nWas my answer good? y/n\n")
                    # if feedback == "y":
                    #     continue
                    # else:
                    #     correction = input(f"\nWhat is the correct answer?\n")
                    #     chat_bot.process_correction(f"{user_input} {correction} [e]")

                    temp += 0.2
                    repetition_penalty += 0.2

    chat()

if __name__ == "__main__":
    main()
