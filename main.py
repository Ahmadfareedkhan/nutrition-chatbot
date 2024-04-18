import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Loads and set environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ['SPEECH_KEY'] = '3ca965cb089e415d85a780e0ce40a3cf'
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

def emergency_assistance(query):
    if not query.strip():
        return "Please provide a query for emergency assistance."
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will act as an emergency assistance provider about nutrition. Whenever the user provides prompt your job is to provide him with better assistance by keeping in mind its an emergency case."},
                {"role": "user", "content": query},
            ]
        )
        response = completion.choices[0].message.content
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

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

# Gradio interfaces
interface1 = gr.Interface(
    fn=chatbot_response,
    inputs=[gr.Textbox(lines=5, label="Input here", placeholder="Type or say your question here..."), gr.Audio(type="filepath", label="Record your question")],
    outputs=[gr.Text(label="Transcription"), gr.Text(label="Response")],
    title="Your AI Nutrition Consultant",
    description="Ask me anything about nutrition"
)

interface2 = gr.Interface(
    fn=emergency_assistance,
    inputs=[gr.Textbox(lines=5, placeholder="Enter your emergency nutrition query here...")],
    outputs=[gr.Text(label="Response")],
    title="Emergency Assistance",
    description="Please provide quick info about your emergency"
)

# Combined interface with tabs
app = gr.TabbedInterface([interface1, interface2], ["Nutrition Consultant", "Emergency Assistance"], title="HealthyBytes: Your AI Nutrition Consultant")

if __name__ == "__main__":
    app.launch(share=False)
