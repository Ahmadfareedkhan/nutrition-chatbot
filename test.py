import pandas as pd
import numpy as np
from langchain_openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Assume your initialization of OpenAI LLM here
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    max_tokens=300,
    temperature=0.5
)


# Define your examples as a list of dictionaries
examples = [
    {"prompt": """
"""}
# {"prompt": """""", "completion": """"""},
# {"prompt": """""", "completion": """"""},
]



def langchain_completion(prompt, job_description):
    """
    This function formats the prompts and completions for use with Langchain's OpenAI wrapper,
    simulating fine-tuning by prepending examples to the prompt.
    """
    # Check if a job description is provided and modify the prompt accordingly
    # if job_description:
    #     prompt_template = f"Given the job description: {job_description}, {prompt}"
    # else:
    #     prompt_template = prompt

    # formatted_prompt = "\n".join([f"Input: {example['prompt']}\nOutput: {example['completion']}" for example in examples]) + f"\nInput: {prompt_template}\nOutput:"

    formatted_prompt = {prompt}

    response = llm.generate(
        prompts=[formatted_prompt]
    )

    if response.generations:
        return response.generations[0][0].text.strip()
    else:
        return "No completion found."





# Example usage
job_description = """
"""
new_prompt = "You are a nutrition consultant AI, capable of providing natural diet plans and emergency assistance based on user inputs"

# completion = langchain_completion_with_projects(new_prompt, job_description)
# print(completion)


def generate_proposal(job_description):
    # Use your existing functionality to generate the proposal
    proposal = langchain_completion(new_prompt, job_description)
    return proposal

# Define the Gradio interface
interface = gr.Interface(fn=generate_proposal,
                         inputs=gr.Textbox(lines=5, placeholder="Type your question here..."),
                         outputs="text",
                         title="HealthyBytes: Your AI Nutrition Consultant",
                         description="Ask me anything about nutrition, diet plans, or emergency assistance. For example, 'What should I eat to stay fit?' or 'What to do in case of a snake bite?'")

# Launch the Gradio app
interface.launch()
