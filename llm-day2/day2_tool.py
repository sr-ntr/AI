from langchain_core.tools import tool


@tool
def calculate_tax(amount: float, tax_rate: float) -> float:
    """Calculate tax for a monetary amount using a decimal tax rate."""
    return amount * tax_rate


print(calculate_tax.invoke({"amount": 1000, "tax_rate": 0.18}))
