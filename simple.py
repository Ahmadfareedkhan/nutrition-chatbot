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
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"





interface2 = gr.Interface(
    fn=emergency_assistance,
    inputs=[gr.Textbox(lines=10, label="Query", placeholder="Enter your emergency nutrition query here...")],
    outputs=[
        gr.Text(lines=10, label="Response"),
    ],
    title="Emergency Assistance",
    description="To better assist you, could you explain what led to this emergency?"
)


# Combined interface with tabs
app = gr.TabbedInterface([interface2], ["Nutrition Consultant", "Emergency Assistance"], title="HealthyBytes: Your AI Nutrition Consultant")

if __name__ == "__main__":
    app.launch(share=False)
