import streamlit as st
import app.utils as utils
import re


def _compact_text(value, max_chars=1200):
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _get_style_digest() -> dict:
    cache_key = "style_digest_cache"
    style_key = "::".join([
        str(st.session_state.get("style", "")),
        str(st.session_state.get("global_rules", "")),
        str(st.session_state.get("guidelines", "")),
        str(st.session_state.get("example", "")),
    ])

    cached = st.session_state.get(cache_key)
    if cached and cached.get("key") == style_key:
        return cached.get("value", {})

    digest = {
        "style": _compact_text(st.session_state.get("style", ""), max_chars=2200),
        "global_rules": _compact_text(st.session_state.get("global_rules", ""), max_chars=1800),
        "guidelines": _compact_text(st.session_state.get("guidelines", ""), max_chars=1200),
        "example": _compact_text(st.session_state.get("example", ""), max_chars=900),
    }

    st.session_state[cache_key] = {"key": style_key, "value": digest}
    return digest


def extract_style(combined_text, debug):
    messages = [
        {"role": "system", "content": st.session_state.locals["llm_instructions"]},
        {"role": "user", "content": st.session_state.locals["training_content"]},
        {"role": "assistant", "content": st.session_state.locals["training_output"]},
        {"role": "user", "content": combined_text},
    ]

    if debug:
        st.write(messages)
    return utils.chat(messages, 0)


def rewrite_content(content_all, max_output_length, debug, context_details=""):
    digest = _get_style_digest()

    system = [
        "You are an expert writer assistant. Rewrite the user input based on the following writing style, global rules, writing guidelines and writing example.\n",
        f"<writingStyle>{digest.get('style', '')}</writingStyle>\n",
        f"<globalRules>{digest.get('global_rules', '')}</globalRules>\n",
        f"<writingGuidelines>{digest.get('guidelines', '')}</writingGuidelines>\n",
        f"<writingExample>{digest.get('example', '')}</writingExample>\n",
        "Make sure to emulate the writing style, global rules, guidelines and example provided above.",
        f"YOU CAN ONLY OUTPUT A MAXIMUM OF {max_output_length} CHARACTERS"
    ]

    if context_details:
        system.insert(
            -1,
            "Use this optional context to shape greetings and situational considerations:\n"
            f"<speechContext>{context_details}</speechContext>\n",
        )

    messages = [
        {"role": "system", "content": "\n".join(system)},
        {"role": "user", "content": content_all},
    ]

    if debug:
        st.write(messages)
    output = utils.chat(messages, 0.7)
    if output:
        return output[:max_output_length]
    return output
