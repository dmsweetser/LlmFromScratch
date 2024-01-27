


from bob_the_bot import BobTheBot
import os

for context_length in [512]:
    for n_layers in [1, 2, 3, 4, 5]:
        for embedding_dim in [16, 64, 128]:
            for lstm_units in [16, 64, 128, 256, 512]:
                for hidden_dim in [16, 64, 128, 256, 512]:
                    for epochs in [1, 50, 100]:
                        for batch_size in [8, 16, 32]:
                            for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                                for dropout in [0.1, 0.2, 0.4, 0.9]:
                                    for recurrent_dropout in [0.1, 0.2, 0.4, 0.9]:

                                        bob_the_bot = BobTheBot(
                                            context_length=context_length, 
                                            learning_rate=learning_rate,
                                            dropout=dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            embedding_dim=embedding_dim, 
                                            lstm_units=lstm_units, 
                                            hidden_dim=hidden_dim, 
                                            n_layers=n_layers, 
                                            epochs=epochs, 
                                            batch_size=batch_size, 
                                            bypass_chat_loop=True)

                                        bob_the_bot.main()
                                        os.system("del model.keras")
                                        os.system("del tokenizer_config.json")
                                        os.system("del training_data.json")