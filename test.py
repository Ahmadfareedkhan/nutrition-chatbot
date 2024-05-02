import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv


# Set an environment variable
load_dotenv('.env')
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Adding custom styling using html and css to improve UI
DESCRIPTION = '''
<div>
<h1 style="text-align: center;">HealthyBytes: Your Personalized AI Nutrition Advisor</h1>
<p>Ask me anything about nutrition. Provide your Gender, Plan Type, Weight and Height for personalized advice.</p>
<p>🔎 For better assistance try to list down your query in detail.</p>
<p>🦕 Below are some featured queries which can help you understand how it works.</p>
</div>
'''

LICENSE = """
<p/>
---
Built by Aneeb
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://enigmaguides.com/wp-content/uploads/2024/03/Nutrition.webp" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;"> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Nutrition Chatbot</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""


css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""


# Function that returns response from Model
def chat_turbo_3_5(message: str, 
                   history: list, 
                   temperature: float, 
                   max_tokens: int
                  ) -> str:
    """
    Generate a response using the OpenAI Turbo 3.5 model.
    """
    conversation = ""
    for user, assistant in history:
        conversation += f"{user}\n{assistant}\n"
    conversation += f"{message}\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs."},
                {"role": "user", "content": conversation},
            ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_text = response.choices[0].message.content
    return output_text


# Gradio block
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Chat Interface')

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=chat_turbo_3_5,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temperature", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=512, 
                      label="Max new tokens", 
                      render=False ),
            ],
        examples=[
            ['What are the main nutrients found in avocados?'],
            ['Can you suggest some high-protein vegetarian meals for building muscle?'],
            ["I have type 2 diabetes. What are some healthy snack options that won't spike my blood sugar?"],
            ['How many calories are there in a grilled chicken salad?'],
            ['I need to prepare a low-carb dinner for four people. Any ideas?']
        ],
        cache_examples=False,
    )
    gr.Markdown(LICENSE)
    
if __name__ == "__main__":
    demo.launch()