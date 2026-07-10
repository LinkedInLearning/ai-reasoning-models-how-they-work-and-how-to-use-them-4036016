"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)

DEFAULT_PROMPT = (
    "A laptop costs $900. "
    "It is discounted by 12%, then a 7.5% sales tax is added to the discounted price. "
    "The customer also uses a $50 gift card after tax. "
    "What is the fina amount due?"
)

# Correct answer: $801.40.

response = client.complete(
    messages=[
        SystemMessage(""""""),
        UserMessage(DEFAULT_PROMPT),
    ],
    model="microsoft/Phi-4-mini-instruct",
    temperature=0.8,
    max_tokens=2048,
    top_p=0.1
)

print(response.choices[0].message.content)
