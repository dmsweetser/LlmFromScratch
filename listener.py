import os
import subprocess
import whisper
import json
import sys
import gtts
import pyaudio
import wave
from bob_the_bot import BobTheBot

def main():
    # Initialize Whisper for STT
    model = whisper.load_model("tiny.en")

    config = {
      "context_length": 128,
      "embedding_dim": 16,
      "lstm_units": 128,
      "hidden_dim": 16,
      "epochs": 40,
      "batch_size": 64,
      "learning_rate": 0.01,
      "dropout": 0.2,
      "recurrent_dropout": 0.2
    }

    chat_bot = BobTheBot(config, True)

    def text_to_speech(text, filename):
        if text == "":
            text = "Sorry - something went wrong."
        tts = gtts.gTTS(text=text, lang='en', tld='com.au')
        tts.save(filename + ".mp3")
        os.system("ffplay -autoexit " + filename + ".mp3")

    def record_audio(filename, duration=10):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = duration

        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

        print("Recording...")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(filename, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

    def listen(audio_file):
        result = model.transcribe(audio_file, fp16=False)
        return result["text"] if result else None

    def listen_for_feedback():
        while True:
            text_to_speech("How can I help?", "output")
            record_audio("input.wav")  # Record audio from microphone
            response = listen("input.wav")
            if response:
                print(f"You said: {response}")
                question = response.strip()

                if "stop listening" in question.lower():
                    input("Press any key to keep listening...")
                    continue
                elif question == "":
                    continue

                # Generate a response using the chatbot
                response = chat_bot.generate_text(chat_bot.end_token, question, chat_bot.model, chat_bot.tokenizer, chat_bot.context_length, num_chars_to_generate=chat_bot.context_length)
                response = response.replace("[e]","")
                # Text-to-speech for the chatbot response
                text_to_speech(response, "output")

                # Text-to-speech for asking feedback
                text_to_speech("Was the result good or bad?", "output")

                # Listen for feedback
                record_audio("input.wav", 2)  # Record feedback from microphone
                feedback = listen("input.wav")
                print(feedback)
                if feedback:
                    if "good" in feedback.lower():
                        print("Back to normal listening...")
                    elif "bad" in feedback.lower():
                        # Text-to-speech for asking correction
                        text_to_speech("Please provide the correct answer:", "output")

                        # Listen for correction
                        record_audio("input.wav")  # Record correction from microphone
                        correction = listen("input.wav")
                        if correction:
                            text_to_speech("Thanks. Let me try and remember that.", "output")
                            chat_bot.process_correction(f"{question} {correction}")
                            print("Processed correction. Back to normal listening...")
                            text_to_speech("Alright. Thanks for that! Let's keep going.", "output")

    listen_for_feedback()               

if __name__ == "__main__":
    main()
