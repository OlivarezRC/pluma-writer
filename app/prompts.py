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

    # Detect speaker from session state for perspective enforcement
    speaker_name = st.session_state.get("speaker_refiner") or ""

    system = [
        "You are an expert writer assistant. Rewrite the user input based on the following writing style, global rules, writing guidelines and writing example.\n",
        f"<writingStyle>{digest.get('style', '')}</writingStyle>\n",
        f"<globalRules>{digest.get('global_rules', '')}</globalRules>\n",
        f"<writingGuidelines>{digest.get('guidelines', '')}</writingGuidelines>\n",
        f"<writingExample>{digest.get('example', '')}</writingExample>\n",
        "Make sure to emulate the writing style, global rules, guidelines and example provided above.",
        f"TARGET WORD COUNT: Your output MUST be approximately {max_output_length // 5} words (roughly {max_output_length} characters), excluding references. "
        f"This is STRICT: the final body must be within ±10-15% of {max_output_length // 5} words "
        f"(between {int((max_output_length // 5) * 0.85)} and {int((max_output_length // 5) * 1.15)} words). Do not produce output outside this range."
    ]

    # Speaker perspective enforcement for speeches
    if speaker_name:
        system.insert(
            -1,
            f"SPEAKER PERSPECTIVE: This speech is delivered by {speaker_name}. "
            "Write from the speaker's OWN first-person perspective ('I', 'we'). "
            f"NEVER refer to {speaker_name} in the third person "
            f"(e.g., WRONG: '{speaker_name} believes...'; RIGHT: 'I believe...').\n",
        )

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
    return output
