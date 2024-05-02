import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import wave

# Loads and set environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
speech_key = os.getenv("speech_key")
os.environ['SPEECH_REGION'] = 'eastus'
client = OpenAI(api_key=api_key)


def recognize_from_microphone(file_info):
    if not file_info:
        return "No audio file received.", ""
    file_path = file_info
    print(f"File path received: {file_path}")  # Verify file path

    # Check file existence
    if not os.path.exists(file_path):
        return f"File not found: {file_path}", ""

    # Configure Azure Speech SDK
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=os.environ['SPEECH_REGION'])
    speech_config.speech_recognition_language = "en-US"
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)

    # Create recognizer and recognize once
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    # Handle recognition result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text, ""
    if result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech could be recognized.", ""
    if result.reason == speechsdk.ResultReason.Canceled:
        return f"Recognition canceled: {result.cancellation_details.reason}", ""

    return "Unexpected error during speech recognition.", ""


def synthesize_speech(text, filename="output.wav"):
    """ Converts text to speech and saves it to a WAV file. """
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=os.environ['SPEECH_REGION'])
    speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{text}] and saved to {filename}")
    else:
        print(f"Failed to synthesize speech for text [{text}]")
    return filename



def chatbot_response(user_input="", gender=None, plan_type=None, weight=None, height=None, audio_input=None):
    transcription, response = "", ""  # Initialize variables for transcription and response
    error_message = ""  # Initialize error_message at the start of the function

    if audio_input:
        transcription, error = recognize_from_microphone(audio_input)
        if error:
            error_message = error  # Capture the error to return it properly
        else:
            user_input = transcription  # Use the transcription if there's no error

    # Check if there's user input or transcription, and there's no error message
    if not user_input.strip() and not transcription.strip() and not error_message:
        error_message = "Please provide audio input or type your question."

    if error_message:
        return error_message, ""  # Return the error with an empty second value

    # Process user_input as before (assuming previous code handled it)
    detailed_input = f"User details - Gender: {gender}, Plan Type: {plan_type}, Weight: {weight} kg, Height: {height} cm. Question: {user_input}"
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs."},
                {"role": "user", "content": detailed_input},
            ]
        )
        response = completion.choices[0].message.content
        if response:
            audio_path = synthesize_speech(response)
            return transcription, response, audio_path  # Return audio path along with text and transcription
    except Exception as e:
        return transcription, f"An error occurred during response generation: {str(e)}"  # Return both values even in case of an exception


def emergency_assistance(query):
    if not query.strip():
        return "Please provide a query for emergency assistance."
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "As an AI serving as an emergency nutrition advisor, your objective is to provide prompt and accurate nutritional guidance in urgent situations. When users present their concerns, you should deliver tailored advice that addresses the critical aspects of their nutritional needs quickly and effectively. Focus on offering clear, practical, and context-specific solutions to ensure their immediate dietary requirements are met."},
                {"role": "user", "content": query},
            ]
        )
        response = completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"
    # After generating response:
    if response:
        audio_path = synthesize_speech(response)
        return response, audio_path  # Return both response text and audio path


# Adjust Gradio interfaces to include audio output
interface1 = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(lines=5, label="Input Here", placeholder="Type or say your question here..."),
        gr.Radio(choices=["Male", "Female", "Other"], label="Gender"),
        gr.Radio(choices=["Weight Gain", "Weight Loss"], label="Plan Type"),
        gr.Number(label="Weight (kg)", info="Enter your weight in kg"),
        gr.Number(label="Height (cm)", info="Enter your height in cm"),
        gr.Audio(type="filepath", label="Record your question")
        ],
    outputs=[
        gr.Text(label="Transcription"),
        gr.Text(lines=10, label="Response"),
        gr.Audio(label="Listen to Response")  # New audio output for the synthesized speech
    ],
    title="Personalized Nutrition AI Advisor",
    description="Ask me anything about nutrition. Provide your Gender, Plan Type, Weight and Height for personalized advice."
)

interface2 = gr.Interface(
    fn=emergency_assistance,
    inputs=[gr.Textbox(lines=10, label="Query", placeholder="Enter your emergency nutrition query here...")],
    outputs=[
        gr.Text(lines=10, label="Response"),
        gr.Audio(label="Listen to Response")  # New audio output for the synthesized speech
    ],
    title="Emergency Assistance",
    description="To better assist you, could you explain what led to this emergency?"
)


# Combined interface with tabs
app = gr.TabbedInterface([interface1, interface2], ["Nutrition Consultant", "Emergency Assistance"], title="HealthyBytes: Your AI Nutrition Consultant")

if __name__ == "__main__":
    app.launch(share=False)
