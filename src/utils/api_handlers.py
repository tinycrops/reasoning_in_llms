"""
API Handlers for Gemini and OpenAI
"""
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables
load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_with_openai(prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=1000):
    """
    Generate text using OpenAI models
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_with_gemini(prompt, model="gemini-pro", temperature=0.7, max_tokens=8192):
    """
    Generate text using Google's Gemini models
    """
    try:
        gemini_model = genai.GenerativeModel(model_name=model)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_with_llm(prompt, provider="openai", model="gpt-4o-mini", temperature=0.7, max_tokens=1000):
    """
    Unified interface for generating text with different LLM providers
    """
    if provider.lower() == "openai":
        return generate_with_openai(prompt, model, temperature, max_tokens)
    elif provider.lower() == "gemini":
        return generate_with_gemini(prompt, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def extract_json_from_response(response_text):
    """
    Attempt to extract a JSON object from the response text
    """
    try:
        # Try to find JSON-like content between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # If not found, try to parse the whole response as JSON
        return json.loads(response_text)
    except Exception as e:
        print(f"JSON extraction error: {e}")
        print(f"Response text: {response_text}")
        return None 