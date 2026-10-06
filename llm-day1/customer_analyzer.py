from google import genai
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()


class CustomerIssue(BaseModel):
    customer_name: str
    issue: str
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


complaints = [
    """
    Hi, I'm Priya. I was charged twice for my subscription this month.
    I've contacted support before and nobody has fixed it. I'm extremely
    frustrated.
    """,
    """
    Hi, I'm Arjun. The app works perfectly and I love the new dashboard.
    """,
    """
    I'm Neha. My payment was declined three times and now my account is locked.
    """,
    """
    This is Rohan. I have a question about whether the premium plan includes API access.
    """,
    """
    I'm Ananya. You charged me twice this month. Please fix this immediately.
    """,
    """
    Rahul here. The product is okay, but the dashboard takes a little too long to load.
    """,
]

for complaint in complaints:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"""
    Analyze this customer complaint.

    Return:
    - customer's name
    - main issue
    - sentiment
    - urgency

    Customer complaint:
    {complaint}
    """,
        config={
            "response_mime_type": "application/json",
            "response_schema": CustomerIssue,
        },
    )

    result = CustomerIssue.model_validate_json(response.text)

    # print(result)
    print()
    print("Customer:", result.customer_name)
    print("Issue:", result.issue)
    print("Sentiment:", result.sentiment)
    print("Urgency:", result.urgency)
