import os
import json
from bob_the_bot import BobTheBot

configs = []
last_executed_index = -1

def create_config():
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
    return config

for epochs in range(1,1000,1): 
    config = create_config()
    config["epochs"] = epochs
    configs.append(config)

try:
   with open('last_executed_index.json', 'r') as file:
        last_executed_index = json.load(file)
except FileNotFoundError:
   pass

def save_last_executed_index(index):
   with open('last_executed_index.json', 'w') as file:
       json.dump(index, file)

def run_next_config():
  global last_executed_index

  if len(configs) > 0 and last_executed_index < len(configs) - 1:
      current_config = configs[last_executed_index + 1]
      bob_the_bot = BobTheBot(current_config, True, "training_data.json", "tokenizer_config.json", "model.keras")
      bob_the_bot.main()
      os.system("del model.keras")
      os.system("del tokenizer_config.json")
      os.system("del training_data.json")
      last_executed_index += 1
      save_last_executed_index(last_executed_index)

if __name__ == "__main__":
    os.system("del model.keras")
    os.system("del tokenizer_config.json")
    os.system("del training_data.json")
    while len(configs) > 0:
        try:
            print(f"Running iteration {last_executed_index} out of {len(configs)}")
            run_next_config()
        except Exception as e:
            print(f"Exception encountered for index {last_executed_index}")
            last_executed_index += 1
            os.system("del model.keras")
            os.system("del tokenizer_config.json")
            os.system("del training_data.json")
            save_last_executed_index(last_executed_index)