import os
from pocketsphinx import LiveSpeech
from gtts import gTTS
import speech_recognition as sr

def main():
    chat_bot = ChatBot()
    recognizer = sr.Recognizer()

    for phrase in speech:
        text = str(phrase)
        print("You said:", text)

        if "hey bob" in text.lower():
            question = text.lower().split("hey bob", 1)[1].strip()

            # Generate a response using the chatbot
            response = chat_bot.generate_text(log_file_path, chat_bot.end_token, question.lower(), chat_bot.model, chat_bot.tokenizer, chat_bot.context_length, num_chars_to_generate=chat_bot.context_length)

            # Text-to-speech for the chatbot response
            tts_result = gTTS(text=response, lang='en')
            tts_result.save("result.mp3")
            os.system("mpg321 result.mp3")

            # Text-to-speech for asking feedback
            tts_feedback = gTTS(text="Was the result good or bad?", lang='en')
            tts_feedback.save("feedback.mp3")
            os.system("mpg321 feedback.mp3")

            # Listening for feedback
            feedback_audio = recognizer.listen(source, timeout=5)
            feedback_text = recognizer.recognize_google(feedback_audio).lower()

            if "good" in feedback_text:
                print("Back to normal listening...")
            elif "bad" in feedback_text:
                # Text-to-speech for asking correction
                tts_correction = gTTS(text="Please provide the correct answer:", lang='en')
                tts_correction.save("correction.mp3")
                os.system("mpg321 correction.mp3")

                # Listening for correction
                correction_audio = recognizer.listen(source, timeout=5)
                correction_text = recognizer.recognize_google(correction_audio).strip()
                chat_bot.process_correction(question, correction_text)
                print("Processed correction. Back to normal listening...")

if __name__ == "__main__":
    main()
