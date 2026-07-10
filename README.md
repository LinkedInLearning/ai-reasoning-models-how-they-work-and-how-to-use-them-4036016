# 03_06 Hands-on: A distilled model in action

This folder is a self-contained Chainlit example for the local Qwen3 models used in this repository. You can copy the folder elsewhere on this computer and run it there.

Run from this folder:

```bash
uv run chainlit run demo_base.py
uv run chainlit run demo_instruct.py
uv run chainlit run demo_reasoning.py
uv run chainlit run demo_distill.py
```

The supported model names are:

- `base`: Qwen3 0.6B base model.
- `instruct`: Qwen3 0.6B (it uses the same weights as the reasoning model but thinking/reasoning is disabled, so it behaves more like a instruction-finetuned model).
- `reasoning`: Qwen3 0.6B reasoning model.
- `distill`: A distilled model based on `base` that I distilled myself using a small math dataset and DeepSeek R1 (more information about the distillation approach can be found [here](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb))

The model files are downloaded into `qwen3` inside this folder by default.

&nbsp;

## Example prompts

> A phone plan charges a fixed monthly fee plus a cost per gigabyte of data. In
> March, Maya used 6 GB and paid \$47. In April, she used 11 GB and paid \$67.
> What is the fixed monthly fee?

(Correct answer: 23)
