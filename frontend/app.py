"""Streamlit frontend — docs/UI_SPEC.md. Talks to the FastAPI backend over HTTP only; holds no NLP logic
itself (per docs/ARCHITECTURE.md §2).
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="TextInsight", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "session_id": None,
        "corpus_info": None,
        "uploaded_file_key": None,
        "chat_history": [],  # list of {"query": str, "response": dict}
        "first_query_done": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session() -> None:
    for key in ("session_id", "corpus_info", "uploaded_file_key", "chat_history", "first_query_done"):
        st.session_state[key] = None if key in ("session_id", "corpus_info", "uploaded_file_key") else (
            [] if key == "chat_history" else False
        )


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------

def _upload_file(file) -> dict | None:
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    data = {"session_id": st.session_state.session_id} if st.session_state.session_id else {}
    try:
        resp = requests.post(f"{BACKEND_URL}/upload", files=files, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
        return None

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        st.error(f"Upload failed: {detail}")
        return None

    return resp.json()


def _submit_query(query: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json={"session_id": st.session_state.session_id, "query": query},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
        return None

    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        st.error(f"Query failed: {detail}")
        return None

    return resp.json()


# ---------------------------------------------------------------------------
# Rendering — per-tool result sections (docs/UI_SPEC.md §4)
# ---------------------------------------------------------------------------

def _render_sentiment(result: dict) -> None:
    distribution = result.get("distribution", {})
    if distribution:
        st.bar_chart(pd.Series(distribution, name="share"))
    per_doc = result.get("per_document", [])
    if per_doc:
        st.dataframe(pd.DataFrame(per_doc), use_container_width=True, hide_index=True)
    if result.get("skipped_count"):
        st.caption(f"{result['skipped_count']} document(s) skipped (empty/too short).")


def _render_classification(result: dict) -> None:
    per_doc = result.get("per_document", [])
    if per_doc:
        labels = [d["label"] for d in per_doc]
        st.bar_chart(pd.Series(labels).value_counts())
        st.dataframe(
            pd.DataFrame([{"id": d["id"], "label": d["label"], "score": d["score"]} for d in per_doc]),
            use_container_width=True,
            hide_index=True,
        )


def _render_summary(result: dict) -> None:
    st.markdown(f"> {result.get('summary', '')}")
    source_ids = result.get("source_document_ids", [])
    st.caption(f"Drawn from {len(source_ids)} document(s): {', '.join(source_ids[:10])}" + (" ..." if len(source_ids) > 10 else ""))
    if result.get("chunked"):
        st.caption("Long input was chunked and summarized in multiple passes.")


def _render_semantic_search(result: dict) -> None:
    results = result.get("results", [])
    if not results:
        st.caption("No matches found.")
        return
    df = pd.DataFrame(results)
    fig = px.bar(df, x="score", y="id", orientation="h", hover_data=["text_excerpt"], labels={"id": "document"})
    fig.update_layout(height=min(400, 60 + 30 * len(df)), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[["id", "score", "text_excerpt"]], use_container_width=True, hide_index=True)


def _render_filter(result: dict) -> None:
    st.caption(f"Filtered to {result.get('count', 0)} document(s).")


def _render_embeddings(result: dict) -> None:
    status = "reused cached index" if result.get("cached") else "built new index"
    st.caption(f"Semantic index: {status} ({result.get('n_vectors', 0)} vectors, {result.get('dim', 0)}-dim).")


def _render_research_evidence(result: dict) -> None:
    """docs/UI_SPEC.md §6."""
    if not result.get("found"):
        st.info("No external evidence found.")
        return
    for item in result.get("evidence", []):
        with st.container(border=True):
            st.markdown(item.get("claim", ""))
            st.markdown(f"[{item.get('source_title', 'source')}]({item.get('source_url', '')})")


def _render_model_recommendation(result: dict) -> None:
    """docs/UI_SPEC.md §4: three clearly separated, visually distinct sections + confidence note."""
    if result.get("degraded"):
        st.warning("LLM synthesis was unavailable — showing the rule-based candidate list without prose rationale.")

    st.markdown(f"### Recommendation: `{result.get('recommendation', '')}`")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**A. Measured on your data**")
        measured = result.get("measured_on_user_data", [])
        if measured:
            st.dataframe(pd.DataFrame(measured), use_container_width=True, hide_index=True)
        else:
            st.caption(result.get("measured_skip_reason") or "No evaluation was run.")

    with col_b:
        st.markdown("**B. External research**")
        evidence = result.get("external_research", [])
        if evidence:
            for item in evidence:
                with st.container(border=True):
                    st.caption(item.get("claim", ""))
                    st.markdown(f"[{item.get('source_title', 'source')}]({item.get('source_url', '')})")
        else:
            st.caption(result.get("research_note") or "No external research available.")

    with col_c:
        st.markdown("**C. System judgment**")
        st.write(result.get("system_judgment", ""))
        for reason in result.get("rationale", []):
            st.caption(f"- {reason}")

    st.info(result.get("confidence_note", ""))
    st.caption(result.get("fine_tune_note", ""))


_RENDERERS = {
    "sentiment_analysis": _render_sentiment,
    "text_classification": _render_classification,
    "summarize_text": _render_summary,
    "semantic_search": _render_semantic_search,
    "filter_documents": _render_filter,
    "generate_embeddings": _render_embeddings,
    "research_models": _render_research_evidence,
    "model_recommendation": _render_model_recommendation,
}


# ---------------------------------------------------------------------------
# Rendering — workflow status + latency panels (docs/UI_SPEC.md §1.2, §7)
# ---------------------------------------------------------------------------

def _render_workflow_panel(plan: list[str], tool_results: dict, latency: dict) -> None:
    ran = set(tool_results.keys())
    parts = []
    for step in plan:
        mark = "✓" if step in ran else "✗"
        step_latency = latency.get(f"tool:{step}")
        tag = f" ({step_latency:.0f}ms)" if step_latency is not None else ""
        parts.append(f"{step} {mark}{tag}")
    st.caption(" → ".join(parts) if parts else "(no tools were run)")


def _render_latency_panel(latency: dict, is_first_query: bool) -> None:
    with st.expander("Latency breakdown", expanded=False):
        if is_first_query:
            st.caption("🧊 Cold start — this run includes first-time model loading, not steady-state speed.")
        if not latency:
            st.caption("No timing data recorded.")
            return
        total = sum(latency.values())
        st.caption(f"Total: {total:.0f} ms")
        df = pd.DataFrame({"step": list(latency.keys()), "ms": list(latency.values())}).sort_values("ms", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("step"))


def render_response(response: dict, is_first_query: bool) -> None:
    if response.get("error") and not response.get("tool_results"):
        st.error(response.get("final_answer") or response["error"])
    elif response.get("error"):
        st.warning("Some steps did not complete — showing what did succeed below.")
        st.write(response.get("final_answer", ""))
    else:
        st.write(response.get("final_answer", ""))

    _render_workflow_panel(response.get("plan", []), response.get("tool_results", {}), response.get("latency", {}))

    tool_results = response.get("tool_results", {})
    for tool_name in response.get("plan", []):
        result = tool_results.get(tool_name)
        if result is None:
            continue
        renderer = _RENDERERS.get(tool_name)
        if renderer is None:
            continue
        with st.expander(tool_name, expanded=(tool_name in ("model_recommendation",))):
            renderer(result)

    _render_latency_panel(response.get("latency", {}), is_first_query)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_init_state()

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload CSV, TXT, or PDF", type=["csv", "txt", "pdf"])

    if uploaded is not None:
        file_key = f"{uploaded.name}:{uploaded.size}"
        if file_key != st.session_state.uploaded_file_key:
            is_replacement = st.session_state.corpus_info is not None
            with st.spinner("Ingesting and profiling..."):
                data = _upload_file(uploaded)
            if data is not None:
                st.session_state.session_id = data["session_id"]
                st.session_state.corpus_info = data
                st.session_state.uploaded_file_key = file_key
                if is_replacement:
                    st.info(f"Replaced the active dataset with **{data['source_filename']}**. Prior analysis no longer applies.")

    if st.session_state.corpus_info:
        info = st.session_state.corpus_info
        profile = info["profile"]
        st.subheader("Session")
        st.caption(f"**{info['source_filename']}** ({info['source_format'].upper()})")
        st.metric("Documents", info["document_count"])
        if info.get("truncated"):
            st.caption("⚠️ Truncated for analysis.")
        st.caption(f"Text column: `{profile.get('text_column') or 'n/a'}`")
        st.caption(f"Detected language: {profile.get('detected_language') or 'unknown'}")
        st.caption(f"Labels present: {'yes' if profile.get('has_labels') else 'no'}")

    st.divider()
    st.subheader("Settings")
    st.text_input(
        "Candidate labels for classification (optional)",
        key="candidate_labels_hint",
        placeholder="billing, technical, delivery, refund",
        help="Mention these in your question and the agent will use them, e.g. \"classify into billing/technical/delivery/refund\".",
    )
    st.checkbox(
        "Request external research when asking about model choice",
        key="research_hint",
        help="Adds a note to your question asking the agent to look up published benchmarks.",
    )

    if st.session_state.session_id and st.button("Clear session"):
        _reset_session()
        st.rerun()

st.title("TextInsight")
st.caption("Upload a dataset and ask questions about it in plain language.")

if st.session_state.corpus_info is None:
    st.info("Upload a CSV, TXT, or PDF file in the sidebar to get started.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            render_response(turn["response"], is_first_query=False)

    example_prompts = (
        "e.g. \"Analyze the sentiment\", \"Why are customers unhappy?\", "
        "\"Find complaints about delayed delivery\", \"Should I use BERT or DistilBERT?\""
    )
    query = st.chat_input(f"Ask a question about your data... ({example_prompts})")

    if query:
        effective_query = query
        if st.session_state.get("candidate_labels_hint"):
            effective_query += f" (candidate labels: {st.session_state.candidate_labels_hint})"
        if st.session_state.get("research_hint"):
            effective_query += " (please include external research if relevant)"

        with st.chat_message("user"):
            st.write(query)

        is_first = not st.session_state.first_query_done
        with st.chat_message("assistant"):
            with st.spinner("Running agent..."):
                response = _submit_query(effective_query)
            if response is not None:
                render_response(response, is_first_query=is_first)

        if response is not None:
            st.session_state.chat_history.append({"query": query, "response": response})
            st.session_state.first_query_done = True
