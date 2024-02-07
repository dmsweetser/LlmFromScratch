import os
import subprocess
import whisper
import json
import sys
import gtts
from bob_the_bot import BobTheBot

def main():
    # Initialize Whisper for STT
    model = whisper.load_model("small.en")

    config = {
      "context_length": 128,
      "embedding_dim": 16,
      "lstm_units": 128,
      "hidden_dim": 16,
      "epochs": 60,
      "batch_size": 64,
      "learning_rate": 0.01,
      "dropout": 0.2,
      "recurrent_dropout": 0.2
    }

    chat_bot = BobTheBot(config, True)

    def text_to_speech(text, filename):
        tts = gtts(text=text, lang='en')
        tts.save(filename + ".mp3")
        os.system("mpg321 " + filename + ".mp3")

    def listen():
        result = model.transcribe("default_source", max_length=10000)
        return result["result"][0]["alternatives"][0]["transcript"].capitalize() if result else None

    def listen_for_feedback():
        input("Press any key to speak")
        response = listen()
        if response:
                print(f"You said: {response}")
                question = response.strip()

                # Generate a response using the chatbot
                response = chat_bot.generate_text(chat_bot.end_token, question, chat_bot.model, chat_bot.tokenizer, chat_bot.context_length, num_chars_to_generate=chat_bot.context_length)

                # Text-to-speech for the chatbot response
                text_to_speech(response, "result")

                # Text-to-speech for asking feedback
                text_to_speech("Was the result good or bad?", "feedback")

                # Listen for feedback
                feedback = listen()

                if feedback:
                    if "good" in feedback:
                        print("Back to normal listening...")
                    elif "bad" in feedback:
                        # Text-to-speech for asking correction
                        text_to_speech("Please provide the correct answer:", "correction")

                        # Listen for correction
                        correction = listen()
                        if correction:
                            chat_bot.process_correction(question, correction)
                            print("Processed correction. Back to normal listening...")

    listen_for_feedback()               

if __name__ == "__main__":
   main()