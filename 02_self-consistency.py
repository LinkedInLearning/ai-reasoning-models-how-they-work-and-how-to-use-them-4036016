import argparse
import os
import re
import time
from collections import Counter

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


PROMPT_FORMAT = (
    "Solve the problem. "
    "At the end, write the final answer on its own line in exactly this format:\n"
    "Final answer: <answer>\n"
    "Do not put any other text on the final-answer line."
)

DEFAULT_PROMPT = DEFAULT_PROMPT + "\n\n" + PROMPT_FORMAT


client = ChatCompletionsClient(
    endpoint="https://models.github.ai/inference",
    credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
)


def call_model(client, model, prompt, sample_id, temperature, top_p, max_tokens):
    print(f"\n[Sample {sample_id}] Sending request...", flush=True)
    start = time.time()

    response = client.complete(
        messages=[
            SystemMessage(
                "You are a careful assistant. Solve the problem clearly. "
                "End your response with a line in exactly this format: Final answer: <answer>"
            ),
            UserMessage(prompt),
        ],
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    elapsed = time.time() - start
    print(f"[Sample {sample_id}] Received response in {elapsed:.1f}s.", flush=True)

    return response.choices[0].message.content


def extract_final_answer(text):
    match = re.search(r"Final answer:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))

    # Fallback: use the last non-empty line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    return normalize_answer(lines[-1])


def normalize_answer(answer):
    answer = answer.strip()
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")
    answer = answer.rstrip(".")
    answer = re.sub(r"\s+", " ", answer)
    return answer.lower()


def self_consistency(prompt, model, num_samples, temperature, top_p, max_tokens):
    responses = []
    extracted_answers = []

    print("\n=== Prompt ===\n", flush=True)
    print(prompt, flush=True)

    for i in range(1, num_samples + 1):
        response = call_model(
            client=client,
            model=model,
            prompt=prompt,
            sample_id=i,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        answer = extract_final_answer(response)

        responses.append(response)
        extracted_answers.append(answer)

        print(f"\n=== Sample {i} response ===\n", flush=True)
        print(response, flush=True)
        print(f"\n[Sample {i}] Extracted final answer: {answer}", flush=True)

    counts = Counter(extracted_answers)
    winner, winner_count = counts.most_common(1)[0]

    print("\n=== Vote counts ===\n", flush=True)
    for answer, count in counts.most_common():
        print(f"{answer}: {count}", flush=True)

    print("\n=== Self-consistency result ===\n", flush=True)
    print(f"Final answer: {winner}", flush=True)
    print(f"Votes: {winner_count}/{num_samples}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt to answer. Uses a default math prompt if omitted.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="GitHub Models model ID.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sampled answers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per sample.",
    )

    args = parser.parse_args()

    self_consistency(
        prompt=args.prompt,
        model=args.model,
        num_samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )