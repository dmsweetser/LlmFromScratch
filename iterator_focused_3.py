import os
import json
from bob_the_bot import BobTheBot

configs = []
last_executed_index = -1

def create_config():
  config = {
      "context_length": None,
      "n_layers": None,
      "embedding_dim": None,
      "lstm_units": None,
      "hidden_dim": None,
      "epochs": None,
      "batch_size": None,
      "learning_rate": None,
      "dropout": None,
      "recurrent_dropout": None
  }
  return config

for context_length in [64]: 
    for embedding_dim in range(16, 128, 16):
        for lstm_units in range(10, 300, 10):
            for hidden_dim in range(10, 300, 10):
                for epochs in [20]:
                    for batch_size in [32]:
                        for learning_rate in [0.01]:
                            for dropout in [0.2]:
                                for recurrent_dropout in [0.2]:
                                    for n_layers in range(5, 50, 5):
                                        for model_variation in [8]:
                                            config = create_config()
                                            config["context_length"] = context_length
                                            config["n_layers"] = n_layers
                                            config["embedding_dim"] = embedding_dim
                                            config["lstm_units"] = lstm_units
                                            config["hidden_dim"] = hidden_dim
                                            config["epochs"] = epochs
                                            config["batch_size"] = batch_size
                                            config["learning_rate"] = learning_rate
                                            config["dropout"] = dropout
                                            config["recurrent_dropout"] = recurrent_dropout
                                            config["model_variation"] = model_variation
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
    print(f"Starting next execution: {last_executed_index + 1} out of {len(configs)}")
    current_config = configs[last_executed_index + 1]
    bob_the_bot = BobTheBot(current_config, True)
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
            run_next_config()
        except Exception as e:
            print(f"Exception encountered for index {last_executed_index}")
            last_executed_index += 1
            os.system("del model.keras")
            os.system("del tokenizer_config.json")
            os.system("del training_data.json")
            save_last_executed_index(last_executed_index)