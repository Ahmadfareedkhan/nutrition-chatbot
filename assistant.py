import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load and set environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ['SPEECH_KEY'] = os.getenv('speech_key')
os.environ['SPEECH_REGION'] = 'eastus'
client = OpenAI(api_key=api_key)

def recognize_from_microphone(file_info):
    if not file_info:
        return "", "No audio file received."
    file_path = file_info
    if not os.path.exists(file_path):
        return "", f"File not found: {file_path}"
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
                {"role": "system", "content": "As an AI serving as an emergency nutrition advisor, your objective is to provide prompt and accurate nutritional guidance in urgent situations. When users present their concerns, you should deliver tailored advice that addresses the critical aspects of their nutritional needs quickly and effectively. Focus on offering clear, practical, and context-specific solutions to ensure their immediate dietary requirements are met."},
                {"role": "user", "content": user_input},
            ]
        )
        response = completion.choices[0].message.content
        return transcription, response
    except Exception as e:
        return transcription, f"An error occurred during response generation: {str(e)}"

    
app = gr.Interface(
    fn=chatbot_response,
    inputs=[gr.Textbox(lines=5, placeholder="Enter your emergency nutrition query here...", label="Input Here"),
            gr.Audio(type="filepath", label="Record your question")],
    outputs=[gr.Text(label="Transcription"), gr.Text(label="Response")],
    title="Emergency Assistance",
    description="To better assist you, could you explain what led to this emergency?"
)

if __name__ == "__main__":
    app.launch(share=False)