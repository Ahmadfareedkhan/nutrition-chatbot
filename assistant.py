import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load and set environment variables
load_dotenv(".env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

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
    
app = gr.Interface(
    fn=emergency_assistance,
    inputs=[gr.Textbox(lines=5, placeholder="Enter your emergency nutrition query here...")],
    outputs=[gr.Text(label="Response")],
    title="Emergency Assistance",
    description="Please provide quick info about your emergency"
)

if __name__ == "__main__":
    app.launch(share=False)