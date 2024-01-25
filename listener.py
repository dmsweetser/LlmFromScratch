import os
import subprocess
import whisper
import srilsh
import json
import sys
from silero import SpeechSynthesizer

def main():
   # Initialize Whisper for STT
   model = whisper.load_model("model_large")

   chat_bot = BobTheBot()

   def text_to_speech(text, filename):
       synthesizer = SpeechSynthesizer()
       synthesizer.say(text, "en-US")
       synthesizer.save_audio("output.wav")
       subprocess.call(["mpg321", "output.wav"])

   def listen():
       result = model.transcribe("default_source", max_length=10000)
       return result["result"][0]["alternatives"][0]["transcript"].capitalize() if result else None

   def listen_for_feedback():
       response = listen()
       if response:
           print(f"You said: {response}")
           if "hey bob" in response:
               question = response.split("hey bob", 1)[1].strip()

               # Generate a response using the chatbot
               response = chat_bot.generate_text(log_file_path, chat_bot.end_token, question, chat_bot.model, chat_bot.tokenizer, chat_bot.context_length, num_chars_to_generate=chat_bot.context_length)

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

if __name__ == "__main__":
   main()