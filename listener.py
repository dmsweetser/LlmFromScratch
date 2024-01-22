import os
from pocketsphinx import LiveSpeech
from gtts import gTTS
import speech_recognition as sr

def process_input(text):
    # Process the input text and return the result
    # Replace this with your actual processing logic
    return f"Processed: {text}"

def main():
    speech = LiveSpeech()
    recognizer = sr.Recognizer()

    for phrase in speech:
        text = str(phrase)
        print("You said:", text)

        if "hey Bob" in text.lower():
            question = text.split("hey Bob", 1)[1].strip()
            result = process_input(question)

            # Text-to-speech for the result
            tts_result = gTTS(text=result, lang='en')
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
                process_correction(question, correction_text)
                print("Processed correction. Back to normal listening...")

def process_correction(question, correction_text):
    # Add logic to handle the correction
    pass

if __name__ == "__main__":
    main()
