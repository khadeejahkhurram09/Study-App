"""
list_deployments.py — discover what AI deployments are actually available.

Run this in the platform terminal to see the real deployment names in your
Azure subscription, then update MODEL_GPT55_DEPLOYMENT (and the others) in
/etc/nixor-lab.env to match.

    python list_deployments.py
"""

import os
import urllib.request
import urllib.error
import json


def list_deployments(endpoint: str, api_key: str, label: str) -> list[dict]:
    """Query the Azure OpenAI deployments list endpoint."""
    if not endpoint or not api_key:
        print(f"  {label}: skipped (endpoint or key not set)")
        return []

    # Both Azure OpenAI and Foundry endpoints expose this REST path.
    url = endpoint.rstrip("/") + "/openai/deployments?api-version=2024-10-21"
    req = urllib.request.Request(url, headers={"api-key": api_key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("data", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print(f"  {label}: HTTP {e.code} — {body}")
        return []
    except Exception as e:
        print(f"  {label}: {e}")
        return []


def print_deployments(label: str, deployments: list[dict]) -> None:
    if not deployments:
        print(f"  {label}: no deployments found\n")
        return
    print(f"  {label}:")
    for d in deployments:
        name = d.get("id") or d.get("model", "?")
        model = d.get("model", {})
        if isinstance(model, dict):
            model_name = f"{model.get('name','?')} {model.get('version','')}"
        else:
            model_name = str(model)
        status = d.get("status", "?")
        print(f"    deployment_name={name!r:30s}  model={model_name.strip()!r}  status={status}")
    print()


def main():
    sep = "─" * 72
    print("=" * 72)
    print("  Azure AI Deployment Discovery")
    print("=" * 72)

    openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    openai_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    foundry_endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
    foundry_key = os.environ.get("AZURE_FOUNDRY_API_KEY", "")

    print("\nChecking AZURE_OPENAI_ENDPOINT …")
    oai_deps = list_deployments(openai_endpoint, openai_key, "azure_openai")
    print_deployments("AZURE_OPENAI_ENDPOINT", oai_deps)

    print("Checking AZURE_FOUNDRY_ENDPOINT …")
    foundry_deps = list_deployments(foundry_endpoint, foundry_key, "azure_foundry")
    print_deployments("AZURE_FOUNDRY_ENDPOINT", foundry_deps)

    print(sep)
    all_names = {d.get("id") or d.get("model", "") for d in oai_deps + foundry_deps}
    print("  All deployment names found:")
    for n in sorted(all_names):
        print(f"    {n!r}")

    print()
    print("  Current env vars:")
    for var in [
        "AZURE_OPENAI_DEPLOYMENT",
        "MODEL_GPT55_DEPLOYMENT",
        "MODEL_GROK43_DEPLOYMENT",
        "MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT",
        "MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT",
    ]:
        print(f"    {var}={os.environ.get(var, '(not set)')!r}")
    print("=" * 72)
    print("  If MODEL_GPT55_DEPLOYMENT is not in the list above, update")
    print("  /etc/nixor-lab.env with the correct deployment name and restart.")
    print("=" * 72)


if __name__ == "__main__":
    main()
