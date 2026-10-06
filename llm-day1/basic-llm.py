from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

prompt = """
You are teaching a beginner data scientist.

Explain embeddings in exactly two sentences.

Rules:
- Sentence 1: define embeddings.
- Sentence 2: explain why they are useful in RAG.
- Use simple language.
- Do not use mathematical notation.
"""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)

print(response.text)
