# 02_06 Hands-on example: Inference-time scaling

This folder compares three ways to answer the same reasoning problem with
models served through GitHub Models:

- a single model call,
- self-consistency through repeated sampling and majority voting,
- self-refinement through an initial answer, a critique, and a revised answer.

The default prompt is a short price-calculation problem. Its correct answer is
`$801.40`. Since model responses are sampled, the exact wording and intermediate
answers can vary between runs.

&nbsp;
## Files

- `01_simple.py` sends the prompt to `microsoft/Phi-4-mini-instruct` once.
- `02_self-consistency.py` samples several answers, extracts each final answer,
  and returns the most frequent result.
- `03_self-refinement.py` uses `microsoft/Phi-4-mini-instruct` for the initial
  and revised answers and `microsoft/Phi-4` for the critique.

&nbsp;
## Setup

The examples require Python and the `azure-ai-inference` package:

```bash
pip install azure-ai-inference
```

The scripts read the GitHub Models access token from the `GITHUB_TOKEN`
environment variable. GitHub Codespaces provides this variable automatically.
For a local environment, create a GitHub personal access token and export it
before running the scripts:

```bash
export GITHUB_TOKEN="YOUR_TOKEN"
```

Do not add the token to the Python files or commit it to the repository.

&nbsp;
## Run the examples

Run the single-call baseline:

```bash
python 01_simple.py
```

Run self-consistency with five sampled answers:

```bash
python 02_self-consistency.py --samples 5
```

The script asks the model to use a consistent final-answer format, normalizes
the extracted answers, counts them, and reports the majority result. You can
also adjust the model and sampling settings:

```bash
python 02_self-consistency.py \
  --samples 7 \
  --temperature 0.8 \
  --top-p 0.9
```

Run the three-step self-refinement workflow:

```bash
python 03_self-refinement.py
```

This script prints the initial answer, the critique, and the refined answer so
that the effect of the critique is visible.

Both multi-step examples accept a custom prompt:

```bash
python 02_self-consistency.py --prompt "Your prompt here"
python 03_self-refinement.py --prompt "Your prompt here"
```

Self-consistency and self-refinement make multiple model calls. They therefore
take longer and use more inference quota than the single-call baseline.
