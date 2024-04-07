import gradio as gr 
from openai import OpenAI
client = OpenAI()
import os

# Set your OpenAI API key here
# openai.api_key = os.getenv("OPENAI_API_KEY")


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)


def chatbot_response(user_input):
    try:
        # Call to OpenAI's API
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs."},
                {"role": "user", "content": user_input},
            ]
        )

        return completion.choices[0].message

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Creating the Gradio interface
interface = gr.Interface(fn=chatbot_response,
                         inputs=gr.Textbox(lines=5, placeholder="Type your question here..."),
                         outputs="text",
                         title="HealthyBytes: Your AI Nutrition Consultant",
                         description="Ask me anything about nutrition, diet plans, or emergency assistance. For example, 'What should I eat to stay fit?' or 'What to do in case of a snake bite?'")

if __name__ == "__main__":
    interface.launch()
