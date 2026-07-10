# 02_02 Hands-on example: Chain-of-thought (CoT) prompting


This folder contains a script that launches a small, local 0.6B Qwen3 model in
a small chat interface. The code is based on:

- my Build a Reasoning Model (From Scratch) repo (https://github.com/rasbt/reasoning-from-scratch),
- Chainlit, https://github.com/chainlit/chainlit

The main file is `qwen3_chat_interface.py`. It starts a local web app with
Chainlit and loads the model code from the `reasoning-from-scratch` Python
package.

&nbsp;
## 1. Install uv

To take care of the project dependencies, this project uses the popular `uv` toool.
`uv` is a Python project and package manager. It creates a virtual environment
for this example and installs the required packages.

On macOS or Linux, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you use Homebrew on macOS, this also works:

```bash
brew install uv
```

After installing, close and reopen your terminal. Then check that `uv` works:

```bash
uv --version
```

The official installation page is here:
<https://docs.astral.sh/uv/getting-started/installation/>

```
uv run chainlit run qwen3_chat_interface.py
```

&nbsp;
## 2. Run the chat interface

For the Qwen3 example, run:

```bash
uv run chainlit run qwen3_chat_interface.py
```

Chainlit will print a local URL in the terminal. It is usually:

```text
http://localhost:8000
```

Open that URL in your browser and send a message in the chat window.

The first run can take longer because Python packages and model files may need
to download.

&nbsp;
## 3. Example prompts

Try out the example prompts:

> Half the value of $3x-9$ is $x+37$. What is the value of $x$?

> Half the value of $3x-9$ is $x+37$. What is the value of $x$? Explain step by
> step
