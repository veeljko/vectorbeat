"""Oznake pesama za jedan upit, sacuvane tokom Streamlit sesije."""

from hashlib import sha256

import streamlit as st


FEEDBACK_LABELS = {"like": "Dopada mi se", "dislike": "Ne dopada mi se"}


def prikazi_oznaku_pesme(doc_id: int, context: tuple) -> None:
    """Prikazi dve iskljucive opcije; ponovni klik uklanja izabranu oznaku.

    Context sadrzi stranu i parametre upita, bez broja/velicine strane.
    Oznake su dostupne kao session_state['ocene_pesama'][context][doc_id]
    sa vrednostima 'like' ili 'dislike'; neoznacene pesme nisu u recniku.
    """
    feedback = st.session_state.setdefault("ocene_pesama", {}).setdefault(context, {})
    context_key = sha256(repr(context).encode("utf-8")).hexdigest()
    choice = st.pills(
        "Oznaka pesme",
        options=list(FEEDBACK_LABELS),
        format_func=FEEDBACK_LABELS.get,
        selection_mode="single",
        required=False,
        default=feedback.get(doc_id),
        key=f"ocena_pesme_{context_key}_{doc_id}",
        label_visibility="collapsed",
        persist_state="session",
    )
    if choice is None:
        feedback.pop(doc_id, None)
    else:
        feedback[doc_id] = choice
