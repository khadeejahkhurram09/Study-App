"""
Your AI App — Nixor AI + Cloud Course
=====================================

This is YOUR app. Right now it's a simple chatbot that talks to a real AI model
(gpt-5.5 via deployment `gpt-5-5` by default) running on Microsoft Azure. Over the
course you'll make it your own:
change its personality, give it a job, add features.

The two lines you'll edit most are marked with  # 👈 EDIT THIS
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load .env when running locally (no-op in the cloud where env vars are set directly).
load_dotenv()

# ---------------------------------------------------------------------------
# 1. Give your app a name and a personality
# ---------------------------------------------------------------------------
APP_TITLE = "BuddyAI"  # 👈 EDIT THIS — what's your app called?

SYSTEM_PROMPT = (  # 👈 EDIT THIS — this is the AI's job description. Be specific!
    "You are a friendly, encouraging planner for CAIE students "
    "You explain ideas simply, use examples, and never just give away answers — "
    "you help the student get there themselves."
)

# ---------------------------------------------------------------------------
# 2. Connect to the AI model running on Azure
#    These values come from your sandbox. Locally they're read from a .env file;
#    in the cloud they're set as App Settings (never written in the code!).
# ---------------------------------------------------------------------------
# gpt-5.5 lives on your Azure OpenAI resource, so the app uses the AZURE_OPENAI_*
# values as a matched set — endpoint, key, and deployment all from the same resource.
_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
client = AzureOpenAI(
    api_key=_api_key,
    azure_endpoint=_endpoint,
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
DEPLOYMENT = os.environ.get("MODEL_GPT55_DEPLOYMENT", "gpt-5-5")


def ask_the_ai(messages: list[dict]) -> str:
    """Send the conversation to Azure and return the AI's reply."""
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        temperature=1,
        max_completion_tokens=600,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 3. The web page (Streamlit turns this Python into a website)
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
st.title(APP_TITLE)
st.caption("Built on Microsoft Azure · Nixor AI + Cloud Course")

# Keep the conversation in memory while the page is open.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Show the conversation so far (skip the hidden system message).
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# A box for the student to type in.
if user_text := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask_the_ai(st.session_state.messages)
        st.write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# IDEAS to try (Sessions 2–4):
#   • Change SYSTEM_PROMPT so the bot becomes a Karachi food guide, a debate
#     coach, a code helper — anything.
#   • Add a st.selectbox so the user can pick a "mode".
#   • Add st.file_uploader + the vision model to describe an uploaded image.
# ---------------------------------------------------------------------------
