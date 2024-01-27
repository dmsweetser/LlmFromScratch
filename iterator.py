


from bob_the_bot import BobTheBot

for context_length in [128, 256, 512, 1024]:
    for embedding_dim in [2, 4, 8, 16, 32, 64, 128]:
        for lstm_units in [2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024]:
            for hidden_dim in [2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024]:
                for n_layers in [1, 2]:
                    for epochs in [1, 2, 5, 10, 20, 30, 40, 50]:
                        for batch_size in [2, 4, 8, 16, 32, 64]:
                            print(f"\n==============================\n"
                                f"Embedding Dim: {embedding_dim}, LSTM Units: {lstm_units},\n"
                                f"Hidden Dim: {hidden_dim}, N_Layers: {n_layers},\n"
                                f"Epochs: {epochs}, Batch Size: {batch_size}\n")
                            bob_the_bot = BobTheBot(context_length=context_length, embedding_dim=embedding_dim, lstm_units=lstm_units, hidden_dim=hidden_dim, n_layers=n_layers, epochs=epochs, batch_size=batch_size, bypass_chat_loop=True)
                            bob_the_bot.main()
                            os.system("rm model.keras")
                            os.system("rm tokenizer_config.json")
                            os.system("rm training_data.json")