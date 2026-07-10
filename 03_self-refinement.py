import argparse
import os
import time

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

DEFAULT_PROMPT = (
    "A laptop costs $900. "
    "It is discounted by 12%, then a 7.5% sales tax is added to the discounted price. "
    "The customer also uses a $50 gift card after tax. "
    "What is the fina amount due?"
)

# Correct answer: $801.40.

client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)

answer_model = "microsoft/Phi-4-mini-instruct"
critique_model = "microsoft/Phi-4"


def call_model(prompt, label, model, temperature=0.8, max_tokens=2048):
    print(f"\n[{label}] Model: {model}", flush=True)
    print(f"[{label}] Sending request...", flush=True)
    start = time.time()

    response = client.complete(
        messages=[
            SystemMessage(
                "You are a careful assistant. Answer clearly and check your work."
            ),
            UserMessage(prompt),
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.1,
    )

    elapsed = time.time() - start
    print(f"[{label}] Received response in {elapsed:.1f}s.", flush=True)

    return response.choices[0].message.content


def self_refine(task):
    print("\n=== Task ===\n", flush=True)
    print(task, flush=True)

    initial_answer = call_model(
        f"""
Answer the following question.

Question:
{task}
""",
        label="Step 1: Initial answer",
        model=answer_model,
        temperature=0.8,
    )

    print("\n=== Initial answer ===\n", flush=True)
    print(initial_answer, flush=True)

    critique = call_model(
        f"""
Review the answer below for correctness.

Question:
{task}

Answer:
{initial_answer}

Identify any mistakes, ambiguities, or missing details. If the answer is already correct,
say so briefly. Do not rewrite the final answer yet.
""",
        label="Step 2: Critique",
        model=critique_model,
        temperature=0.2,
    )

    print("\n=== Critique ===\n", flush=True)
    print(critique, flush=True)

    refined_answer = call_model(
        f"""
Revise the answer using the critique.

Question:
{task}

Initial answer:
{initial_answer}

Critique:
{critique}

Return only the improved final answer.
""",
        label="Step 3: Refined answer",
        model=answer_model,
        temperature=0.2,
    )

    print("\n=== Refined answer ===\n", flush=True)
    print(refined_answer, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to answer. Uses a default prompt if omitted.",
    )
    args = parser.parse_args()

    self_refine(args.prompt)