import gradio as gr
import openai
import os

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

def chatbot_response(user_input):
    try:
        # Call to OpenAI's API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs."},
                {"role": "user", "content": user_input},
            ]
        )
        # Extracting the AI response
        return response.choices[0].message.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Creating the Gradio interface
interface = gr.Interface(fn=chatbot_response,
                         input=gr.inputs.Textbox(lines=5, placeholder="Type your question here..."),
                         output="text",
                         title="HealthyBytes: Your AI Nutrition Consultant",
                         description="Ask me anything about nutrition, diet plans, or emergency assistance. For example, 'What should I eat to stay fit?' or 'What to do in case of a snake bite?'")

if __name__ == "__main__":
    interface.launch()
