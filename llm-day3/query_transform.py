# If user query is bad (doesn't contain any keyword), keyword retrieval won't help
# Use query transformation, and produce RRF rankings using user query based rankings and transformed query based rankings
# User query preserves the actual wording and intent of the user
# Transformed query helps retrieve evidence using different terminology

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model=os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    ),
    google_api_key=os.environ["GEMINI_API_KEY"],
)


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        description="A search-friendly version of the original question."
    )
    preserves_intent: bool = Field(
        description="Whether the rewritten query preserves the original intent."
    )
    reason: str = Field(
        description="Briefly explain whether the meaning was preserved."
    )


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Rewrite the user's question into a search-friendly query.

        Preserve the original intent, conditions, dates, quantities,
        negations, and uncertainty.

        Do not introduce assumptions or treat an unverified claim
        as an established fact.

        If you cannot safely preserve the intent, return the original
        question and set preserves_intent to false.
        """,
        ),
        ("human", "{question}"),
    ]
)

rewrite_chain = rewrite_prompt | model.with_structured_output(QueryRewrite)


def safe_rewrite(question: str) -> QueryRewrite:
    result = rewrite_chain.invoke({"question": question})

    if not result.preserves_intent or not result.rewritten_query.strip():
        return QueryRewrite(
            rewritten_query=question,
            preserves_intent=False,
            reason="Using the original question because the rewrite was unsafe.",
        )

    return result
