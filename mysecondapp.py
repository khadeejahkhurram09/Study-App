"""
Model Playground & Eval Lab — Nixor AI + Cloud Course (Day 4 stretch)
=====================================================================

Two learning exercises in one app:

  1. ⚔️  Head-to-Head — send ONE prompt to ALL four sandbox models at once
         (in parallel) and read the answers side by side. Teaching point:
         style, speed, depth and cost differ even on the same question.

  2. 📊  Benchmark Eval — run a small fixed benchmark across all four models,
         auto-grade every answer, and rank the models on a leaderboard.
         Teaching point: this is (a tiny version of) how model evals actually
         work — a fixed question set, a grader, and a score you can compare.
         Two grading modes are included so students see both paradigms:
            • Objective  — deterministic rules (exact / keyword / numeric match)
            • LLM-as-judge — one model grades the others' answers

Where each model lives (unchanged from the starter):
  • GPT-5.5            → your Azure OpenAI resource  (AZURE_OPENAI_* env vars)
  • Grok-4.3           → your Foundry resource        (AZURE_FOUNDRY_* env vars)
  • DeepSeek-V4-Pro    → your Foundry resource
  • Mistral-Medium-3.5 → your Foundry resource

Run with:  streamlit run model_playground.py
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration / model registry  (same structure as the starter app)
# --------------------------------------------------------------------------- #
OPENAI_EP = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
FOUNDRY_EP = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
FOUNDRY_KEY = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

# label -> (deployment name, endpoint, api key)
MODELS = {
    "GPT-5.5": (os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5"), OPENAI_EP, OPENAI_KEY),
    "Grok-4.3": (os.environ.get("MODEL_GROK43_DEPLOYMENT", "xai-grok43"), FOUNDRY_EP, FOUNDRY_KEY),
    "DeepSeek-V4-Pro": (os.environ.get("MODEL_DEEPSEEK_V4_PRO_DEPLOYMENT", "ds-v4pro"), FOUNDRY_EP, FOUNDRY_KEY),
    "Mistral-Medium-3.5": (os.environ.get("MODEL_MISTRAL_MEDIUM_35_DEPLOYMENT", "mstr-med35"), FOUNDRY_EP, FOUNDRY_KEY),
}

# Azure retail (list) prices in USD per 1,000,000 tokens, pulled from the Azure
# Retail Prices API for these deployments (GlobalStandard token meters).
# NOTE: actual billed rate can differ under sponsorship/contract credits, discounts,
# or taxes; cached / batch / provisioned meters are priced separately. Still
# editable live in the Benchmark tab.
DEFAULT_PRICING = {
    "GPT-5.5": {"input": 1.25, "output": 10.00},
    "Grok-4.3": {"input": 1.25, "output": 2.50},
    "DeepSeek-V4-Pro": {"input": 1.74, "output": 3.48},
    "Mistral-Medium-3.5": {"input": 1.50, "output": 7.50},
}
DEFAULT_USD_PKR = 278.0  # rough rate; editable in the UI

# --------------------------------------------------------------------------- #
# Pure helpers — NO Streamlit calls in this section, so it is unit-testable.
# --------------------------------------------------------------------------- #


def model_cost(prompt_tokens, completion_tokens, price):
    """Cost in USD given token counts and a {'input','output'} price per 1M tokens."""
    pt = prompt_tokens or 0
    ct = completion_tokens or 0
    return (pt / 1_000_000) * price["input"] + (ct / 1_000_000) * price["output"]


def extract_numbers(text):
    """Pull every number out of a string as floats. '17 x 23 = 391' -> [17, 23, 391]."""
    if not text:
        return []
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)  # drop thousands separators
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", cleaned)]


def _normalize(s):
    """Lowercase, trim, and strip surrounding punctuation/quotes for exact matching."""
    return (s or "").strip().lower().strip(".,!?;:'\"`​ ").strip()


@dataclass
class BenchItem:
    id: str
    category: str
    prompt: str
    mode: str            # "numeric" | "keywords" | "exact"
    answers: list        # acceptable answers (numbers for numeric, strings otherwise)
    tol: float = 1e-6    # tolerance for numeric comparison
    note: str = ""       # optional teaching note


# A deliberately small, mixed benchmark. Each item is auto-gradable.
BENCHMARK = [
    BenchItem("q1", "Arithmetic",
              "What is 17 multiplied by 23? Reply with just the number.",
              "numeric", [391]),
    BenchItem("q2", "Geography",
              "What is the capital city of Australia? Answer in one word.",
              "keywords", ["canberra"],
              note="A common trap — many guess Sydney."),
    BenchItem("q3", "Reasoning",
              "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
              "than the ball. How much does the ball cost?",
              "numeric", [0.05, 5],
              note="The classic cognitive-reflection trap; the wrong answer is 0.10."),
    BenchItem("q4", "Instruction-following",
              "Reply with exactly one word and nothing else: BANANA",
              "exact", ["banana"]),
    BenchItem("q5", "Literature",
              "Who wrote the play 'Romeo and Juliet'? A surname is enough.",
              "keywords", ["shakespeare"]),
    BenchItem("q6", "Counting",
              "How many letters are in the word 'strawberry'? Reply with just the number.",
              "numeric", [10],
              note="Letter-counting is a known weak spot for tokenized models."),
    BenchItem("q7", "Science",
              "Which gas do plants absorb from the air during photosynthesis?",
              "keywords", ["carbon dioxide", "co2"]),
    BenchItem("q8", "Word problem",
              "A train travels 60 km in 1.5 hours. What is its average speed in "
              "km/h? Reply with just the number.",
              "numeric", [40]),
    BenchItem("q9", "Urdu literature",
              "Who wrote Bagh o Buhar?",
              "keywords", ["mir amman", "mir amman dihlavi", "meer ummun"]),
    BenchItem("q10", "Urdu literature",
              "Who wrote this sher: آج بھی قافلۂ عشق رواں ہے کہ جو تھا, وہی میل اور وہی سنگ نشاں ہے کہ جو تھا?",
              "keywords", ["firaq", "firaq gorakhpuri", "فراق گورکھپوری"]),
    BenchItem("q11", "Chemistry",
              "What is the chemical symbol for carbon? Reply with one word.",
              "keywords", ["c"]),
    BenchItem("q12", "Arithmetic",
              "What is 144 divided by 12? Reply with just the number.",
              "numeric", [12]),
    BenchItem("q13", "Physics",
          "What is the SI unit of power? Reply with just one word.",
          "keywords", ["watt", "watts"]),
    BenchItem("q14", "Math",
          "What is the derivative of 5x with respect to x? Reply with just the number.",
          "numeric", [5]),
     BenchItem("q15", "Explanation",
           "In one sentence, explain why the sky appears blue.",
           "keywords", ["scatter"],
           note="Objective grading is weak here — a good LLM-judge case."),
]


def keyword_hit(answer, keywords):
    """Whole-word keyword match — avoids 'au' matching inside 'sauce'."""
    low = answer.lower()
    return any(re.search(rf"\b{re.escape(str(k).lower())}\b", low) for k in keywords)
def grade(item: BenchItem, answer: str) -> bool:
    """Deterministic grader. Returns True if `answer` satisfies the item."""
    if answer is None:
        return False
    if item.mode == "numeric":
        nums = extract_numbers(answer)
        return any(abs(n - a) <= item.tol for n in nums for a in item.answers)
    if item.mode == "keywords":
        return keyword_hit(answer, item.answers)
    if item.mode == "exact":
        norm = _normalize(answer)
        return any(norm == _normalize(str(a)) for a in item.answers)
    raise ValueError(f"unknown grading mode: {item.mode}")


def build_client(endpoint, key, api_version):
    """Isolated so tests can monkeypatch the network layer."""
    return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)


def call_model(label, prompt, system=None, max_tokens=400, temperature=1.0,
               api_version=API_VERSION):
    """Call one model. Always returns a dict; never raises.

    Keys: label, ok, answer, error, latency, prompt_tokens, completion_tokens.
    """
    deployment, endpoint, key = MODELS[label]
    out = {"label": label, "ok": False, "answer": "", "error": None,
           "latency": None, "prompt_tokens": None, "completion_tokens": None}
    if not endpoint or not key:
        out["error"] = "endpoint or key not set in environment"
        return out

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    try:
        client = build_client(endpoint, key, api_version)
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        out["latency"] = time.perf_counter() - t0
        out["answer"] = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            out["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            out["completion_tokens"] = getattr(usage, "completion_tokens", None)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 — surface any provider error to the UI
        out["latency"] = time.perf_counter() - t0
        out["error"] = str(exc)
    return out


def run_head_to_head(prompt, labels, **kw):
    """Fire all selected models concurrently. Returns {label: result_dict}."""
    results = {}
    if not labels:
        return results
    with ThreadPoolExecutor(max_workers=len(labels)) as ex:
        futs = {ex.submit(call_model, l, prompt, **kw): l for l in labels}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return results


JUDGE_SYSTEM = (
    "You are a strict grader. You are given a QUESTION, a REFERENCE answer, and "
    "a CANDIDATE answer. Decide whether the candidate is correct. Respond ONLY "
    'with a JSON object: {"correct": true|false, "reason": "<short reason>"}.'
)


def parse_judge_verdict(raw):
    """Pull a boolean correctness verdict out of the judge's reply, robustly."""
    if not raw:
        return False
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return bool(data.get("correct"))
    except (json.JSONDecodeError, AttributeError):
        pass
    low = raw.lower()
    return ("true" in low or "correct" in low) and "incorrect" not in low


def judge_answer(judge_label, item: BenchItem, candidate, **kw):
    """Use one model to grade another's answer. Returns (correct: bool, raw: str)."""
    ref = item.answers[0]
    user = (f"QUESTION: {item.prompt}\nREFERENCE: {ref}\nCANDIDATE: {candidate}\n"
            "Is the candidate correct?")
    # The judge uses its own fixed generation settings — strip any max_tokens /
    # temperature inherited from the benchmark run so they don't collide below.
    kw.pop("max_tokens", None)
    kw.pop("temperature", None)
    res = call_model(judge_label, user, system=JUDGE_SYSTEM, max_tokens=120,
                     temperature=1.0, **kw)
    if not res["ok"]:
        return False, res.get("error", "")
    return parse_judge_verdict(res["answer"]), res["answer"]


def run_benchmark(labels, grading_mode="objective", judge_label=None,
                  progress=None, **kw):
    """Run every benchmark item against every selected model and grade it.

    Returns {label: {results: [...], correct, total, accuracy, avg_latency}}.
    `progress` (optional) is called with a float 0..1 as work completes.
    """
    per_model = {l: {"results": [], "correct": 0, "total": 0, "latencies": [],
                     "prompt_tokens": 0, "completion_tokens": 0} for l in labels}
    total_steps = max(1, len(BENCHMARK) * len(labels))
    step = 0

    for item in BENCHMARK:
        answers = run_head_to_head(item.prompt, labels, **kw)
        for l in labels:
            r = answers[l]
            ans = r.get("answer", "") or ""
            if not r["ok"]:
                correct = False
            elif grading_mode == "judge" and judge_label:
                correct, _ = judge_answer(judge_label, item, ans, **kw)
            else:
                correct = grade(item, ans)

            per_model[l]["results"].append({
                "id": item.id, "category": item.category, "prompt": item.prompt,
                "answer": ans, "ok": r["ok"], "correct": bool(correct),
                "latency": r.get("latency"), "error": r.get("error"),
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
            })
            per_model[l]["total"] += 1
            per_model[l]["correct"] += int(bool(correct))
            per_model[l]["prompt_tokens"] += (r.get("prompt_tokens") or 0)
            per_model[l]["completion_tokens"] += (r.get("completion_tokens") or 0)
            if r.get("latency") is not None:
                per_model[l]["latencies"].append(r["latency"])
            step += 1
            if progress:
                progress(step / total_steps)

    for l in labels:
        m = per_model[l]
        m["accuracy"] = m["correct"] / m["total"] if m["total"] else 0.0
        m["avg_latency"] = (sum(m["latencies"]) / len(m["latencies"])
                            if m["latencies"] else None)
    return per_model


def leaderboard_rows(per_model, pricing=None, fx=None):
    """Flatten benchmark results into sortable leaderboard rows (with cost if priced)."""
    rows = []
    for label, m in per_model.items():
        row = {
            "Model": label,
            "Accuracy": round(m["accuracy"], 3),
            "Correct": f"{m['correct']}/{m['total']}",
            "Avg latency (s)": round(m["avg_latency"], 2) if m["avg_latency"] else None,
            "Out tokens": m.get("completion_tokens", 0),
        }
        if pricing and label in pricing:
            cost = model_cost(m.get("prompt_tokens", 0),
                              m.get("completion_tokens", 0), pricing[label])
            row["Cost (USD)"] = round(cost, 6)
            row["Cost/correct (USD)"] = round(cost / m["correct"], 6) if m["correct"] else None
            if fx:
                row["Cost (PKR)"] = round(cost * fx, 3)
        rows.append(row)
    # primary: accuracy desc; tiebreak: cheaper first
    rows.sort(key=lambda r: (r["Accuracy"], -(r.get("Cost (USD)") or 0.0)), reverse=True)
    return rows


def detailed_rows(per_model, pricing=None):
    """One row per (model, question) — the full audit trail for CSV export."""
    out = []
    for label, m in per_model.items():
        price = pricing.get(label) if pricing else None
        for r in m["results"]:
            pt, ct = r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0
            row = {
                "Model": label, "Question": r["id"], "Category": r["category"],
                "Correct": r["correct"], "OK": r["ok"],
                "Latency (s)": round(r["latency"], 3) if r.get("latency") else None,
                "In tokens": pt, "Out tokens": ct,
            }
            if price:
                row["Cost (USD)"] = round(model_cost(pt, ct, price), 6)
            row["Answer"] = r["answer"]
            out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def _render_result_card(container, res):
    container.markdown(f"**{res['label']}**")
    if not res["ok"]:
        container.error(res["error"] or "failed")
        return
    container.write(res["answer"])
    bits = []
    if res["latency"] is not None:
        bits.append(f"⏱️ {res['latency']:.2f}s")
    if res["completion_tokens"] is not None:
        bits.append(f"🔢 {res['completion_tokens']} out-tokens")
    if bits:
        container.caption("  •  ".join(bits))


def render_head_to_head(settings):
    st.subheader("⚔️ Head-to-Head")
    st.caption("One prompt → all selected models at once. Compare style, speed, depth.")

    prompt = st.text_area(
        "Your prompt",
        "Explain what a neural network is in exactly 3 sentences, for a 16-year-old.",
        key="h2h_prompt",
    )

    if st.button("Run on all models", key="h2h_run"):
        labels = settings["labels"]
        if not labels:
            st.warning("Select at least one model in the sidebar.")
            return
        with st.spinner(f"Asking {len(labels)} models in parallel…"):
            results = run_head_to_head(
                prompt, labels,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
            )
        st.session_state["h2h_results"] = results

    results = st.session_state.get("h2h_results")
    if results:
        ok = [r for r in results.values() if r["ok"] and r["latency"] is not None]
        if ok:
            fastest = min(ok, key=lambda r: r["latency"])
            st.success(f"Fastest to respond: **{fastest['label']}** "
                       f"({fastest['latency']:.2f}s)")
        labels = [l for l in settings["labels"] if l in results]
        for i in range(0, len(labels), 2):  # two cards per row
            cols = st.columns(2)
            for col, label in zip(cols, labels[i:i + 2]):
                with col.container(border=True):
                    _render_result_card(col, results[label])


def render_benchmark(settings):
    st.subheader("📊 Benchmark Eval")
    st.caption("Run a fixed question set across the models and rank them. "
               "This is a miniature of how real model evals work.")

    with st.expander(f"See the {len(BENCHMARK)} benchmark questions"):
        st.dataframe(
            pd.DataFrame([{
                "id": b.id, "category": b.category, "prompt": b.prompt,
                "mode": b.mode, "accepted answer(s)": ", ".join(map(str, b.answers)),
            } for b in BENCHMARK]),
            hide_index=True, width="stretch",
        )

    mode_label = "LLM-as-judge" if settings["grading_mode"] == "judge" else "Objective rules"
    st.info(f"Grading mode: **{mode_label}**"
            + (f" · judge = **{settings['judge_label']}**"
               if settings["grading_mode"] == "judge" else ""))

    if st.button("Run benchmark", key="bench_run"):
        labels = settings["labels"]
        if not labels:
            st.warning("Select at least one model in the sidebar.")
            return
        bar = st.progress(0.0, text="Running benchmark…")
        per_model = run_benchmark(
            labels,
            grading_mode=settings["grading_mode"],
            judge_label=settings["judge_label"],
            progress=lambda f: bar.progress(f, text=f"Running benchmark… {int(f*100)}%"),
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"],
        )
        bar.empty()
        st.session_state["bench_results"] = per_model

    per_model = st.session_state.get("bench_results")
    if per_model:
        active = list(per_model.keys())

        # --- editable pricing (live-recomputes cost without re-running) ---
        st.markdown("#### 💵 Pricing")
        st.caption("Azure retail list prices (USD per 1M tokens), from the Retail "
                   "Prices API. Editable — your billed rate may differ under "
                   "credits/discounts. Cost updates instantly; no re-run needed.")
        price_base = pd.DataFrame([{
            "Model": l,
            "Input $/1M": DEFAULT_PRICING.get(l, {}).get("input", 1.0),
            "Output $/1M": DEFAULT_PRICING.get(l, {}).get("output", 3.0),
        } for l in active])
        edited = st.data_editor(price_base, hide_index=True, width="stretch",
                                disabled=["Model"], key="pricing_editor")
        pricing = {row["Model"]: {"input": float(row["Input $/1M"]),
                                  "output": float(row["Output $/1M"])}
                   for _, row in edited.iterrows()}
        fx = st.number_input("USD → PKR rate (for the PKR column)",
                             value=DEFAULT_USD_PKR, min_value=1.0, step=1.0)

        # --- leaderboard with cost ---
        rows = leaderboard_rows(per_model, pricing=pricing, fx=fx)
        lb_df = pd.DataFrame(rows)
        st.markdown("#### 🏆 Leaderboard")
        st.caption("Ranked by accuracy, then by lower cost. **Cost/correct** is the "
                   "real 'best value' metric — cheapest way to get a right answer.")
        st.dataframe(lb_df, hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        c1.markdown("**Accuracy**")
        c1.bar_chart(lb_df.set_index("Model")[["Accuracy"]])
        if "Cost (USD)" in lb_df.columns:
            c2.markdown("**Value frontier** — up & left is better (accurate + cheap)")
            c2.scatter_chart(lb_df, x="Cost (USD)", y="Accuracy")

        # --- CSV export ---
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Leaderboard CSV", lb_df.to_csv(index=False).encode("utf-8"),
            "leaderboard.csv", "text/csv", key="dl_lb")
        det_df = pd.DataFrame(detailed_rows(per_model, pricing=pricing))
        d2.download_button(
            "⬇️ Full results CSV", det_df.to_csv(index=False).encode("utf-8"),
            "benchmark_results.csv", "text/csv", key="dl_det")

        st.markdown("#### 🔍 Per-question breakdown")
        any_label = next(iter(per_model))
        n_q = len(per_model[any_label]["results"])
        for i in range(n_q):
            item = BENCHMARK[i]
            verdicts = []
            for label, m in per_model.items():
                r = m["results"][i]
                verdicts.append(("✅" if r["correct"] else "❌") + " " + label)
            with st.expander(f"{item.id} · {item.category} — " + "  ".join(verdicts)):
                st.write(f"**Prompt:** {item.prompt}")
                if item.note:
                    st.caption("💡 " + item.note)
                for label, m in per_model.items():
                    r = m["results"][i]
                    mark = "✅" if r["correct"] else "❌"
                    ans = r["answer"] if r["ok"] else f"_error: {r['error']}_"
                    st.markdown(f"{mark} **{label}:** {ans}")


def sidebar_settings():
    st.sidebar.header("⚙️ Settings")
    labels = st.sidebar.multiselect(
        "Models to include", list(MODELS.keys()), default=list(MODELS.keys()),
    )
    max_tokens = st.sidebar.slider("Max output tokens", 50, 1000, 400, 50)
    temperature = st.sidebar.slider(
        "Temperature", 0.0, 2.0, 1.0, 0.1,
        help="Some gpt-5 / reasoning deployments only accept the default of 1.0.",
    )
    grading_mode = st.sidebar.radio(
        "Benchmark grading", ["objective", "judge"],
        format_func=lambda m: "Objective rules" if m == "objective" else "LLM-as-judge",
        help="Objective = deterministic match. Judge = one model grades the others.",
    )
    judge_label = None
    if grading_mode == "judge":
        judge_label = st.sidebar.selectbox("Judge model", list(MODELS.keys()))

    missing = [l for l in labels if not (MODELS[l][1] and MODELS[l][2])]
    if missing:
        st.sidebar.warning("Missing endpoint/key for: " + ", ".join(missing))

    return {"labels": labels, "max_tokens": max_tokens, "temperature": temperature,
            "grading_mode": grading_mode, "judge_label": judge_label}


def main():
    st.set_page_config(page_title="Model Playground & Eval Lab", page_icon="🧪",
                       layout="wide")
    st.title("🧪 KKK")
    st.caption("Nixor AI + Cloud Course — compare four models head-to-head and "
               "benchmark them.")
    settings = sidebar_settings()
    tab1, tab2 = st.tabs(["⚔️ Head-to-Head", "📊 Benchmark Eval"])
    with tab1:
        render_head_to_head(settings)
    with tab2:
        render_benchmark(settings)


if __name__ == "__main__":
    main()