import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", google_api_key=os.environ["GEMINI_API_KEY"]
)

prompt = ChatPromptTemplate(
    [
        ("system", "You are a concise AI/ML tutor"),
        ("human", "Explain {topic} in exactly {sentences} sentences"),
    ]
)

chain = prompt | model

response = chain.invoke({"topic": "embeddings", "sentences": 2})

print(response.text)
