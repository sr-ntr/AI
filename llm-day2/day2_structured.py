import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class CustomerIssue(BaseModel):
    customer_name: str
    issue: str
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", google_api_key=os.environ["GEMINI_API_KEY"]
)

structured_model = model.with_structured_output(CustomerIssue)

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You analyze the customer complaints. Extract the requested information",
        ),
        ("human", "Analyze this complaint : \n\n{complaint}"),
    ]
)

chain = prompt | structured_model

result = chain.invoke(
    {
        "complaint": """
    Hi, I'm Priya. I was charged twice for my subscription this month.
    I've contacted support before and nobody has fixed it.
    I'm extremely frustrated.
    """
    }
)

print(result)
print(type(result))
print(result.customer_name)
print(result.issue)
print(result.sentiment)
print(result.urgency)
