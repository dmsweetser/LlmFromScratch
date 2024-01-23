import os
from pocketsphinx import LiveSpeech
from gtts import gTTS
import speech_recognition as sr

def main():
    # Initialize LiveSpeech and source
    speech = LiveSpeech()
    recognizer = sr.Recognizer()
    source = sr.Microphone()

    bob_the_bot = BobTheBot()

    def text_to_speech(text, filename):
        tts_result = gTTS(text=text, lang='en')
        tts_result.save(filename)
        os.system(f"ffmpeg -i {filename} -hide_banner -loglevel panic -acodec libmp3lame -aq 4 {filename}.mp3")
        os.system(f"start {filename}.mp3")

    def listen_for_feedback():
        feedback_audio = recognizer.listen(source, timeout=5)
        return recognizer.recognize_google(feedback_audio).lower()

    def listen_for_correction():
        correction_audio = recognizer.listen(source, timeout=5)
        return recognizer.recognize_google(correction_audio).strip()

    for phrase in speech:
        text = str(phrase)
        print("You said:", text)

        if "hey bob" in text.lower():
            question = text.lower().split("hey bob", 1)[1].strip()

            # Generate a response using the chatbot
            response = bob_the_bot.generate_text(log_file_path, bob_the_bot.end_token, question.lower(), bob_the_bot.model, bob_the_bot.tokenizer, bob_the_bot.context_length, num_chars_to_generate=bob_the_bot.context_length)

            # Text-to-speech for the chatbot response
            text_to_speech(response, "result")

            # Text-to-speech for asking feedback
            text_to_speech("Was the result good or bad?", "feedback")

            # Listening for feedback
            feedback_text = listen_for_feedback()

            if "good" in feedback_text:
                print("Back to normal listening...")
            elif "bad" in feedback_text:
                # Text-to-speech for asking correction
                text_to_speech("Please provide the correct answer:", "correction")

                # Listening for correction
                correction_text = listen_for_correction()
                bob_the_bot.process_correction(question, correction_text)
                print("Processed correction. Back to normal listening...")

if __name__ == "__main__":
    main()
