# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Reasoning Model From Scratch"
# Code: https://github.com/rasbt/reasoning-from-scratch


import os
import re
from pathlib import Path

import torch
import chainlit
from chainlit.config import config as chainlit_config

from reasoning_from_scratch.ch02 import (
    get_device,
)
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
from reasoning_from_scratch.ch03 import (
    load_model_and_tokenizer,
    load_tokenizer_only,
    render_prompt,
)
from reasoning_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B


# ============================================================
# EDIT ME: Simple configuration
# ============================================================
WHICH_MODEL = "base"  # "reasoning" for base model
MAX_NEW_TOKENS = 38912
LOCAL_DIR = "qwen3"
USE_RENDER_PROMPT = os.getenv("USE_RENDER_PROMPT", "1") != "0"
# Set CHECKPOINT_PATH to load a custom .pth checkpoint instead of the
# default weights in LOCAL_DIR. Keep WHICH_MODEL aligned with the tokenizer
# that checkpoint expects; chapter 8 distillation checkpoints use "reasoning".
# Terminal example:
#   CHECKPOINT_PATH=/absolute/path/to/model.pth \
#   uv run chainlit run qwen3_chat_interface.py
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
COMPILE = False
# ============================================================


DEVICE = get_device()
CH04_EXAMPLE_RAW_PROMPT = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)
CH04_STEP_BY_STEP_SUFFIX = " \n\nExplain step by step."


def enable_chainlit_latex_rendering():
    chainlit_config.features.latex = True


def normalize_latex_delimiters(text):
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    parts = re.split(r"(\$\$.*?\$\$|\$[^$\n]*?\$)", text, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"(?<![\w\\])(?:\\|/)boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
            r"$\\boxed{\1}$",
            parts[i],
        )
    return "".join(parts)


def prepare_prompt(raw_prompt, explain_step_by_step=False):
    prompt = render_prompt(raw_prompt) if USE_RENDER_PROMPT else raw_prompt
    if explain_step_by_step:
        prompt += CH04_STEP_BY_STEP_SUFFIX
    return prompt


enable_chainlit_latex_rendering()


def resolve_user_prompt(message_text):
    stripped = message_text.strip()
    if stripped == "/ch04":
        return prepare_prompt(CH04_EXAMPLE_RAW_PROMPT)
    if stripped == "/ch04-cot":
        return prepare_prompt(CH04_EXAMPLE_RAW_PROMPT, explain_step_by_step=True)
    return prepare_prompt(message_text)


def load_app_model_and_tokenizer():
    if CHECKPOINT_PATH is None:
        return load_model_and_tokenizer(
            which_model=WHICH_MODEL,
            device=DEVICE,
            use_compile=COMPILE,
            local_dir=LOCAL_DIR,
        )

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    tokenizer = load_tokenizer_only(which_model=WHICH_MODEL, local_dir=LOCAL_DIR)
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)

    if COMPILE:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


MODEL = None
TOKENIZER = None
EOS_TOKEN_IDS = None


def ensure_model_loaded():
    global MODEL, TOKENIZER, EOS_TOKEN_IDS
    if MODEL is not None:
        return

    MODEL, TOKENIZER = load_app_model_and_tokenizer()

    # Even though the official TOKENIZER.eos_token_id is either <|im_end|> (reasoning)
    # or <|endoftext|> (base), some custom checkpoints emit both.
    EOS_TOKEN_IDS = (
        TOKENIZER.encode("<|im_end|>")[0],
        TOKENIZER.encode("<|endoftext|>")[0],
    )


@chainlit.on_chat_start
async def on_start():
    ensure_model_loaded()
    chainlit.user_session.set("history", [])
    chainlit.user_session.get("history").append(
        {"role": "system", "content": "You are a helpful assistant."}
    )


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    # 1) Encode input
    ensure_model_loaded()
    prompt = resolve_user_prompt(message.content)
    input_ids = TOKENIZER.encode(prompt)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    # 2) Start an outgoing message we can stream into
    out_msg = chainlit.Message(content="")
    await out_msg.send()

    # 3) Stream generation
    generated_pieces = []
    for tok in generate_text_basic_stream_cache(
        model=MODEL,
        token_ids=input_ids_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
        token_id = tok.squeeze(0).item()
        if token_id in EOS_TOKEN_IDS:
            break
        piece = TOKENIZER.decode([token_id])
        generated_pieces.append(piece)
        await out_msg.stream_token(piece)

    # 4) Finalize the streamed message
    out_msg.content = normalize_latex_delimiters("".join(generated_pieces))
    await out_msg.update()
