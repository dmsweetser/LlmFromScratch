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
      "recurrent_dropout": None,
      "temperature": 1.0,
      "repetition_penalty": 1.0
  }
  return config


        #     "context_length": 256,
        #     "embedding_dim": 256,
        #     "lstm_units": 256,
        #     "hidden_dim": 4096, 
        #     "n_layers": 4,
        #     "epochs": 10,
        #     "batch_size": 32,
        #     "learning_rate": 0.01,
        #     "dropout": 0.2,
        #     "recurrent_dropout": 0.2,
        #     "temperature": 1.0,
        #     "repetition_penalty": 1.0
        # }

for context_length in [64]: 
    for embedding_dim in range(32,512,32):
        for lstm_units in range(32,512,32):
            for hidden_dim in range(32,8192,32):
                for epochs in [10]:
                    for batch_size in [32]:
                        for learning_rate in [0.01]:
                            for dropout in [0.2]:
                                for recurrent_dropout in [0.2]:
                                    for n_layers in range(1,30,1):
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
            run_next_config()
        except Exception as e:
            print(f"Exception encountered for index {last_executed_index}")
            last_executed_index += 1
            os.system("del model.keras")
            os.system("del tokenizer_config.json")
            os.system("del training_data.json")
            save_last_executed_index(last_executed_index)