import os
import time
import math
import json
from dotenv import load_dotenv
import groq

class LLMClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = groq.Client(api_key=self.api_key)
        self.model = "llama3-70b-8192"

    def complete(self, prompt, max_tokens=1000, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None

def create_structured_prompt(text, question):
    prompt = f"""
# Detailed Content Analysis Report

## Task
Your role is to perform content analysis and classification. 
Your task is to carefully evaluate the input text, determine its underlying intent, 
and assign it to the most appropriate category as prompted by the question.

## Input Text
{text}

## Question
{question}

## Response Format
Classification: [The appropriate category or classification]
Explanation: [A brief explanation of your reasoning]

## Analysis
"""
    return prompt

def classify_with_confidence(client, text, categories, confidence_threshold=0.8):
    prompt = f"""
Classify the following text into exactly one of these categories: {', '.join(categories)}.
Respond in the following JSON format exactly:

{{
    "Sentiment": "[one of: {', '.join(categories)}]",
    "Confidence": [a decimal number between 0 and 1],
    "Explanation": "[A brief explanation]"
}}

Text to classify:
{text}
"""
    try:
        response = client.client.chat.completions.create(
            model=client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0,
        )
        completion = response.choices[0].message.content
        output = json.loads(completion)
        
        if output.get("Confidence", 0) >= confidence_threshold:
            return output
        else:
            output["Sentiment"] = "uncertain"
            output["Explanation"] = "Confidence below threshold"
            return output
    except Exception as e:
        print(f"Error in classification: {e}")
        return None

def compare_prompt_strategies(client, texts, categories):
    strategies = {
        "basic": lambda text: f"""
Classify the following text into one of these categories: {', '.join(categories)}.

Respond in JSON format with the keys:
    "Sentiment": one of the provided categories,
    "Confidence": a decimal number between 0 and 1,
    "Explanation": a brief explanation of your reasoning.

Text: {text}
""",
        "structured": lambda text: f"""
Please classify the text below into one of the following categories: {', '.join(categories)}.

Respond in JSON format with the keys:
    "Sentiment": one of the provided categories,
    "Confidence": a decimal number between 0 and 1,
    "Explanation": a brief explanation of your reasoning.

Text: {text}
""",
        "few_shot": lambda text: f"""
Below are some examples of text classification:

Example 1:
Text: "The service was excellent and the food was delightful."
Response: {{
    "Sentiment": "Positive",
    "Confidence": 0.95,
    "Explanation": "The words 'excellent' and 'delightful' indicate a strong positive sentiment."
}}

Example 2:
Text: "I was disappointed by the long wait and poor service."
Response: {{
    "Sentiment": "Negative",
    "Confidence": 0.90,
    "Explanation": "The words 'disappointed' and 'poor service' convey dissatisfaction."
}}

Example 3:
Text: "The experience was okay, nothing too special."
Response: {{
    "Sentiment": "Neutral",
    "Confidence": 0.75,
    "Explanation": "The phrase 'nothing too special' suggests an average or neutral sentiment."
}}

Now, classify the following text:

Text: "{text}"

Respond in JSON format with:
    "Sentiment": one of the provided categories,
    "Confidence": a decimal number between 0 and 1,
    "Explanation": a brief explanation of your reasoning.
"""
    }

    results = {}
    for strategy_name, prompt_func in strategies.items():
        strategy_results = []
        for text in texts:
            prompt = prompt_func(text)
            response = client.complete(prompt, max_tokens=500, temperature=0.5)
            strategy_results.append(response)
        results[strategy_name] = strategy_results

    return results

if __name__ == "__main__":
    client = LLMClient()
    sample_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "The service was terrible and I will not be coming back.",
        "The experience was just okay, nothing extraordinary."
    ]
    categories = ["Positive", "Negative", "Neutral"]
    strategy_results = compare_prompt_strategies(client, sample_texts, categories)

    for strategy, results in strategy_results.items():
        print(f"Strategy: {strategy}")
        for result in results:
            print("Result:", result)
            print("-" * 50)
