import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load and set environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ['SPEECH_KEY'] = '3ca965cb089e415d85a780e0ce40a3cf'
os.environ['SPEECH_REGION'] = 'eastus'
client = OpenAI(api_key=api_key)

def recognize_from_microphone(file_info):
    if not file_info:
        return "", "No audio file received."

    file_path = file_info  # Assuming file_info is the correct file path

    # Verify the file exists before trying to open it
    if not os.path.exists(file_path):
        return "", f"File not found: {file_path}"

    # Initialize speech configuration
    speech_config = speechsdk.SpeechConfig(subscription=os.environ['SPEECH_KEY'], region=os.environ['SPEECH_REGION'])
    speech_config.speech_recognition_language = "en-US"

    try:
        audio_config = speechsdk.audio.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = speech_recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text, ""
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "", "No speech could be recognized."
        elif result.reason == speechsdk.ResultReason.Canceled:
            return "", f"Speech Recognition canceled: {result.cancellation_details.reason}."
    except Exception as e:
        return "", f"Error during speech recognition: {str(e)}"

    return "", "Unexpected error during speech recognition."

def chatbot_response(user_input="", audio_input=None):
    transcription, error = recognize_from_microphone(audio_input) if audio_input else ("", "")
    
    if transcription:
        user_input = transcription
    
    if not user_input.strip():
        return error or "Please provide some input or speak into the microphone.", ""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs."},
                {"role": "user", "content": user_input},
            ]
        )
        response = completion.choices[0].message.content
        return transcription, response
    except Exception as e:
        return transcription, f"An error occurred during response generation: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=chatbot_response,
    inputs=[gr.Textbox(lines=5, placeholder="Type or say your question here..."), gr.Audio(type="filepath", label="Record your question")],
    outputs=[gr.Text(label="Transcription"), gr.Text(label="Response")],
    title="HealthyBytes: Your AI Nutrition Consultant",
    description="Ask me anything about nutrition, diet plans, or emergency assistance. Speak or type your question."
)

if __name__ == "__main__":
    interface.launch(share=True)
