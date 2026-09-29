import json
import os
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Multi-Agent Orchestrator", page_icon="🤖", layout="wide")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ---------- Agents ----------
AGENTS = {
    "DocEditor": (
        "You are Agent-DocEditor, a specialist sub-agent for document editing, proofreading, "
        "formatting, reports and summaries. Complete the task fully and return polished "
        "Markdown. Do not ask follow-up questions; make sensible assumptions and state them briefly."
    ),
    "WebDesigner": (
        "You are Agent-WebDesigner, a specialist sub-agent for frontend UI/UX. Produce layouts, "
        "wireframes (as structured text) and clean responsive HTML with Tailwind CSS. Return code "
        "in fenced code blocks and briefly explain key design decisions. Do not ask follow-up questions."
    ),
    "MultiTasker": (
        "You are Agent-MultiTasker, a specialist sub-agent for research, web assistance, quick "
        "troubleshooting and API/automation design. Give concise, actionable output (steps, "
        "examples). Clearly flag anything you cannot verify. Do not ask follow-up questions."
    ),
}

ROUTER_PROMPT = (
    "You are the Central Master Agent (Orchestrator). Read the user request and decide which "
    "sub-agents are needed. Available agents:\n"
    "- DocEditor: document editing, formatting, reports, summaries\n"
    "- WebDesigner: frontend UI/UX, layouts, HTML/Tailwind, wireframes\n"
    "- MultiTasker: research, web assistance, troubleshooting, API/automation\n\n"
    "Use only the agents that are really needed (1 to 3). Reply ONLY with JSON in this shape:\n"
    '{"summary": "one-line request summary", '
    '"plan": [{"agent": "DocEditor|WebDesigner|MultiTasker", "task": "specific task for that agent"}]}'
)

ASSEMBLE_PROMPT = (
    "You are the Central Master Agent (Orchestrator). Combine the sub-agent outputs into one "
    "answer in this exact Markdown format:\n"
    "## Request Summary\n## Routing Plan\n"
    "## Delegation Details (for each agent: Assigned Agent, Task Breakdown, Expected Output)\n"
    "## Final Assembled Execution\n"
    "In the final section, merge the agents' work into one clear, ready-to-use result. "
    "Keep code blocks intact."
)


# ---------- Helpers ----------
def get_api_key(label: str, env_name: str) -> str:
    key = st.sidebar.text_input(label, type="password")
    if key:
        return key.strip()
    if os.getenv(env_name):
        return os.environ[env_name]
    try:
        return st.secrets[env_name]
    except Exception:
        return ""


def chat(client, model, system, user, history=None, temperature=0.3, json_mode=False):
    messages = [{"role": "system", "content": system}]
    messages += history or []
    messages.append({"role": "user", "content": user})
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        try:
            kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception:
            kwargs.pop("response_format", None)  # some providers don't support JSON mode
    return client.chat.completions.create(**kwargs).choices[0].message.content


def clean_json(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    return raw.strip()


def recent_history(limit=4, max_chars=1500):
    msgs = st.session_state.messages[:-1][-limit:]  # exclude the current prompt
    return [{"role": m["role"], "content": m["content"][:max_chars]} for m in msgs]


def route(client, model, prompt, history):
    raw = chat(client, model, ROUTER_PROMPT, prompt, history, temperature=0.1, json_mode=True)
    try:
        data = json.loads(clean_json(raw))
        plan = [p for p in data.get("plan", []) if p.get("agent") in AGENTS]
        summary = data.get("summary", prompt[:100])
    except (json.JSONDecodeError, AttributeError):
        plan, summary = [], prompt[:100]
    if not plan:  # fallback if routing fails
        plan = [{"agent": "MultiTasker", "task": prompt}]
    return summary, plan


# ---------- UI ----------
st.title("🤖 Multi-Agent Orchestrator")
st.caption("Master Agent routes your task to DocEditor, WebDesigner and MultiTasker.")

st.sidebar.header("Settings")
provider = st.sidebar.selectbox("Provider", ["Gemini (free tier)", "OpenAI"])

if provider == "OpenAI":
    model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], help="gpt-4o-mini is cheaper")
    api_key = get_api_key("OpenAI API Key", "OPENAI_API_KEY")
    base_url = None
else:
    model = st.sidebar.text_input(
        "Model", value="gemini-2.5-flash", key="gemini_model",
        help="If this model name stops working, check aistudio.google.com for the current name.",
    ).strip()
    api_key = get_api_key("Gemini API Key", "GEMINI_API_KEY")
    base_url = GEMINI_BASE_URL

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Apna task yahan likhein..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            st.error("Sidebar mein apni API Key paste karein.")
            st.stop()

        client = OpenAI(api_key=api_key, base_url=base_url)
        try:
            with st.status("Master Orchestrator coordinating agents...", expanded=True) as status:
                history = recent_history()

                st.write("1. Orchestrator is planning the routing...")
                summary, plan = route(client, model, prompt, history)
                st.write("Selected agents: " + ", ".join(p["agent"] for p in plan))

                st.write("2. Agents are working in parallel...")
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {
                        p["agent"]: pool.submit(
                            chat, client, model, AGENTS[p["agent"]], p["task"], history
                        )
                        for p in plan
                    }
                    outputs = {name: f.result() for name, f in futures.items()}

                st.write("3. Assembling final output...")
                assemble_input = f"User Request: {prompt}\nSummary: {summary}\n\n" + "\n\n".join(
                    f"### Agent-{p['agent']} (task: {p['task']})\n{outputs[p['agent']]}" for p in plan
                )
                final = chat(client, model, ASSEMBLE_PROMPT, assemble_input, temperature=0.2)
                status.update(label="Complete!", state="complete", expanded=False)

            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})
        except Exception as e:
            st.error(f"Error: {e}")
            st.caption(
                "Check the API key and model name. On Gemini's free tier, a 'rate limit' / 429 "
                "error means wait a minute and try again."
            )
