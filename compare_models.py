"""
compare_models.py — send the same prompt to all 4 sandbox models and print the results.

This is an evaluation (eval) exercise: you can see how each model's style, depth,
and accuracy differs on identical input. Change PROMPT to whatever you want to test.

Run inside the platform terminal:
    python compare_models.py

All environment variables are already injected by the platform — no .env needed.
"""

import os
import textwrap
import time
from openai import AzureOpenAI

# ── What do you want to ask? ────────────────────────────────────────────────
PROMPT = (
    "Explain what a neural network is in exactly 3 sentences, "
    "suitable for a 16-year-old student."
)

# ── The 4 sandbox models ─────────────────────────────────────────────────────
MODELS = [
    {
        "id": "gpt-5.5",
        "label": "GPT-5.5",
        # gpt-5.5 lives on the Azure OpenAI resource (not Foundry), so it uses the
        # AZURE_OPENAI_* endpoint/key together with its deployment name.
        "deployment": os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
    },
    {
        "id": "grok-4.3",
        "label": "Grok-4.3",
        "deployment": os.environ.get("MODEL_GROK43_DEPLOYMENT", "xai-grok43"),
        "endpoint": os.environ.get("AZURE_FOUNDRY_ENDPOINT", ""),
        "api_key": os.environ.get("AZURE_FOUNDRY_API_KEY", ""),
    },
    {
        "id": "DeepSeek-V4-Pro",
        "label": "DeepSeek-V4-Pro",
        "deployment": os.environ.get("MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT", "ds-v4pro"),
        "endpoint": os.environ.get("AZURE_FOUNDRY_ENDPOINT", ""),
        "api_key": os.environ.get("AZURE_FOUNDRY_API_KEY", ""),
    },
    {
        "id": "mistral-medium-3-5",
        "label": "mistral-medium-3-5",
        "deployment": os.environ.get("MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT", "mstr-med35"),
        "endpoint": os.environ.get("AZURE_FOUNDRY_ENDPOINT", ""),
        "api_key": os.environ.get("AZURE_FOUNDRY_API_KEY", ""),
    },
]

API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

WIDTH = 78


def divider(char="─"):
    return char * WIDTH


def ask(model: dict, prompt: str) -> tuple[str, float]:
    """Call the model and return (reply_text, elapsed_seconds)."""
    client = AzureOpenAI(
        api_key=model["api_key"],
        azure_endpoint=model["endpoint"],
        api_version=API_VERSION,
    )
    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=model["deployment"],
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=300,
    )
    elapsed = time.monotonic() - t0
    reply = response.choices[0].message.content or ""
    return reply.strip(), elapsed


def main():
    print(divider("═"))
    print("  MODEL COMPARISON — 4 Azure AI models")
    print(divider("═"))
    print(f"\nPrompt: {PROMPT}\n")

    results = []
    for m in MODELS:
        if not m["endpoint"] or not m["api_key"]:
            results.append((m["label"], "⚠  endpoint/key not set in environment", 0.0))
            continue
        print(f"  Asking {m['label']}…", end="", flush=True)
        try:
            reply, elapsed = ask(m, PROMPT)
            results.append((m["label"], reply, elapsed))
            print(f" {elapsed:.1f}s")
        except Exception as exc:
            results.append((m["label"], f"ERROR: {exc}", 0.0))
            print(" error")

    print()
    for label, reply, elapsed in results:
        print(divider())
        timing = f"  ({elapsed:.1f}s)" if elapsed else ""
        print(f"  {label}{timing}")
        print(divider())
        wrapped = textwrap.fill(reply, width=WIDTH - 4, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)
        print()

    print(divider("═"))
    print("  Done. Tip: change PROMPT at the top of this file to test something else.")
    print(divider("═"))


if __name__ == "__main__":
    main()
