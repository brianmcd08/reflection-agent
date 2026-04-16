import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for "
            "the user's tweet. Always provide detailed recommendations, including requests for length, "
            "virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. If the user provides critique, "
            "respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatAnthropic(
    model_name=os.environ["ANTHROPIC_MODEL"],
    timeout=30,
    stop=[],
    max_tokens_to_sample=512,
)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
