# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Reasoning Model From Scratch"
# Code: https://github.com/rasbt/reasoning-from-scratch


import os
import re
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import (
    generate_text_basic_stream_cache,
    get_device,
)
from reasoning_from_scratch.ch03 import load_model_and_tokenizer, load_tokenizer_only
from reasoning_from_scratch.qwen3 import (
    download_qwen3_distill_checkpoints,
    Qwen3Model,
    QWEN_CONFIG_06_B,
)

try:
    import chainlit
    from chainlit.config import config as chainlit_config
except ModuleNotFoundError:
    class _MissingChainlit:
        class Message:
            pass

        @staticmethod
        def on_chat_start(func):
            return func

        @staticmethod
        def on_message(func):
            return func

    chainlit = _MissingChainlit()
    chainlit_config = None


SUPPORTED_MODELS = ("base", "instruct", "reasoning", "distill")
DEFAULT_MODEL = "reasoning"
DEFAULT_LOCAL_DIR = "qwen3"
DEFAULT_MAX_NEW_TOKENS = 38912
DEFAULT_DISTILL_TYPE = "deepseek_r1"
DEFAULT_DISTILL_STEP = "06682"
SELECTED_MODEL = "reasoning"


class ModelConfig:
    def __init__(
        self,
        model_name,
        source_model,
        add_thinking,
        checkpoint_kind,
        checkpoint_path,
        distill_type,
        distill_step,
        local_dir,
        max_new_tokens,
        use_compile,
    ):
        self.model_name = model_name
        self.source_model = source_model
        self.add_thinking = add_thinking
        self.checkpoint_kind = checkpoint_kind
        self.checkpoint_path = checkpoint_path
        self.distill_type = distill_type
        self.distill_step = distill_step
        self.local_dir = local_dir
        self.max_new_tokens = max_new_tokens
        self.use_compile = use_compile


class AppRuntime:
    def __init__(self, model, tokenizer, device, config, eos_token_ids):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.eos_token_ids = eos_token_ids


RUNTIME = None


def enable_chainlit_latex_rendering():
    if chainlit_config is not None:
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


enable_chainlit_latex_rendering()


def normalize_model_name(model_name):
    normalized = (model_name or DEFAULT_MODEL).strip().lower()
    if normalized not in SUPPORTED_MODELS:
        options = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {options}")
    return normalized


def get_model_config(model_name=None):
    selected = normalize_model_name(
        model_name or os.getenv("CHAINLIT_DEMO_MODEL", SELECTED_MODEL)
    )

    source_model = selected
    add_thinking = False
    checkpoint_kind = None
    checkpoint_path = None

    if selected == "instruct":
        source_model = "reasoning"
    elif selected == "reasoning":
        add_thinking = True
    elif selected == "distill":
        source_model = "reasoning"
        add_thinking = True
        checkpoint_kind = "distill"
        checkpoint_path_text = os.getenv("CHECKPOINT_PATH")
        if checkpoint_path_text:
            checkpoint_path = Path(checkpoint_path_text).expanduser()

    return ModelConfig(
        model_name=selected,
        source_model=source_model,
        add_thinking=add_thinking,
        checkpoint_kind=checkpoint_kind,
        checkpoint_path=checkpoint_path,
        distill_type=os.getenv("CH08_DISTILL_TYPE", DEFAULT_DISTILL_TYPE),
        distill_step=os.getenv("CH08_DISTILL_STEP", DEFAULT_DISTILL_STEP),
        local_dir=Path(os.getenv("QWEN3_LOCAL_DIR", DEFAULT_LOCAL_DIR)),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)),
        use_compile=os.getenv("COMPILE", "0").lower() in {"1", "true", "yes"},
    )


def get_eos_token_ids(tokenizer):
    eos_token_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)

    for token in ("<|im_end|>", "<|endoftext|>"):
        token_ids = tokenizer.encode(token, chat_wrapped=False)
        if token_ids:
            eos_token_ids.add(token_ids[0])

    return eos_token_ids


def load_checkpoint_model_and_tokenizer(config, device):
    if config.checkpoint_path is None:
        checkpoint_path = download_qwen3_distill_checkpoints(
            distill_type=config.distill_type,
            step=config.distill_step,
            out_dir=config.local_dir,
        )
    else:
        checkpoint_path = config.checkpoint_path

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    tokenizer = load_tokenizer_only(
        which_model=config.source_model,
        local_dir=config.local_dir,
    )
    tokenizer.add_thinking = config.add_thinking

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(device)

    if config.use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    model.eval()
    return model, tokenizer


def load_selected_model_and_tokenizer(config, device):
    if config.checkpoint_kind == "distill":
        return load_checkpoint_model_and_tokenizer(config, device)

    model, tokenizer = load_model_and_tokenizer(
        which_model=config.source_model,
        device=device,
        use_compile=config.use_compile,
        local_dir=config.local_dir,
    )
    tokenizer.add_thinking = config.add_thinking
    model.eval()
    return model, tokenizer


def get_runtime():
    global RUNTIME

    if RUNTIME is None:
        config = get_model_config()
        device = get_device()
        model, tokenizer = load_selected_model_and_tokenizer(config, device)
        RUNTIME = AppRuntime(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            eos_token_ids=get_eos_token_ids(tokenizer),
        )

    return RUNTIME


def trim_input_tensor(input_ids_tensor, context_len, max_new_tokens):
    if max_new_tokens >= context_len:
        raise ValueError("MAX_NEW_TOKENS must be smaller than the model context length")

    keep_len = max(1, context_len - max_new_tokens)
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]
    return input_ids_tensor


def build_chat_prompt(history, add_thinking):
    parts = []
    for message in history:
        parts.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        )

    parts.append("<|im_start|>assistant")
    if add_thinking:
        parts.append("\n")
    else:
        parts.append("\n<think>\n\n</think>\n\n")
    return "".join(parts)


def encode_prompt(runtime, history, latest_message):
    if runtime.config.model_name == "base":
        input_ids = runtime.tokenizer.encode(latest_message, chat_wrapped=False)
    else:
        prompt = build_chat_prompt(history, add_thinking=runtime.config.add_thinking)
        input_ids = runtime.tokenizer.encode(prompt, chat_wrapped=False)

    input_ids_tensor = torch.tensor(input_ids, device=runtime.device).unsqueeze(0)
    return trim_input_tensor(
        input_ids_tensor=input_ids_tensor,
        context_len=runtime.model.cfg["context_length"],
        max_new_tokens=runtime.config.max_new_tokens,
    )


@chainlit.on_chat_start
async def on_start():
    get_runtime()
    chainlit.user_session.set(
        "history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@chainlit.on_message
async def main(message):
    runtime = get_runtime()
    history = chainlit.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    input_ids_tensor = encode_prompt(
        runtime=runtime,
        history=history,
        latest_message=message.content,
    )

    out_msg = chainlit.Message(content="")
    await out_msg.send()

    for token in generate_text_basic_stream_cache(
        model=runtime.model,
        token_ids=input_ids_tensor,
        max_new_tokens=runtime.config.max_new_tokens,
    ):
        token_id = token.squeeze(0).item()
        if token_id in runtime.eos_token_ids:
            break
        await out_msg.stream_token(runtime.tokenizer.decode([token_id]))

    out_msg.content = normalize_latex_delimiters(out_msg.content)
    await out_msg.update()

    history.append({"role": "assistant", "content": out_msg.content})
    chainlit.user_session.set("history", history)
