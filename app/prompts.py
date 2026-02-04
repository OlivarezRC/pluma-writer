import streamlit as st
import app.utils as utils


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


def rewrite_content(content_all, max_output_length, debug):
    system = [
        "You are an expert writer assistant. Rewrite the user input based on the following writing style, global rules, writing guidelines and writing example.\n",
        f"<writingStyle>{st.session_state.style}</writingStyle>\n",
        f"<globalRules>{st.session_state.global_rules}</globalRules>\n",
        f"<writingGuidelines>{st.session_state.guidelines}</writingGuidelines>\n",
        f"<writingExample>{st.session_state.example}</writingExample>\n",
        "Make sure to emulate the writing style, global rules, guidelines and example provided above.",
        f"YOU CAN ONLY OUTPUT A MAXIMUM OF {max_output_length} WORDS"
    ]

    messages = [
        {"role": "system", "content": "\n".join(system)},
        {"role": "user", "content": content_all},
    ]

    if debug:
        st.write(messages)
    return utils.chat(messages, 0.7)
