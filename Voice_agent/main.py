import os
from dotenv import load_dotenv
from agent import run_agent  # Importing the brain you just built
import speech_recognition as sr
import pyttsx3

# Initialize the "Mouth" (Text-to-Speech)
engine = pyttsx3.init()
engine.setProperty('rate', 175) # Set speed of speech

def speak(text):
    print(f"Agent: {text}")
    engine.say(text)
    engine.runAndWait()

# Initialize the "Ears" (Speech-to-Text)
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("\n[Listening... Speak now]")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except Exception as e:
            return None

def start_voice_assistant():
    load_dotenv()
    speak("Voice Agent is active. How can I help you today?")

    while True:
        user_text = listen()
        
        if user_text:
            print(f"You said: {user_text}")
            
            # Exit command
            if "exit" in user_text.lower() or "stop" in user_text.lower():
                speak("Shutting down. Goodbye!")
                break
            
            # Get response from your NVIDIA RAG system
            try:
                response = run_agent(user_text)
                speak(response)
            except Exception as e:
                print(f"Error: {e}")
                speak("I'm sorry, I encountered an error connecting to the cloud.")
        else:
            print("...")

if __name__ == "__main__":
    start_voice_assistant()