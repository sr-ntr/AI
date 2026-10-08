import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


@tool
def calculate_tax(amount: float, tax_rate: float) -> float:
    """Calculate tax for a monetary amount using a decimal tax rate."""
    return amount * tax_rate


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", google_api_key=os.environ["GEMINI_API_KEY"]
)

model_with_tools = model.bind_tools([calculate_tax])

response = model_with_tools.invoke("What is 18% tax on INR 1000?")


print(response)
