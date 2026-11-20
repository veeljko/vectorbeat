"""Build and refresh a playlist from the current search's song judgments."""

from hashlib import sha256

import streamlit as st

from mrvectorbeat.search.ranking.rocchio.playlist import build_feedback_playlist
from mrvectorbeat.textprocessing.text_processor import TextProcessor


def _display_words(terms, source_ids, catalog):
    """Use original lyric forms rather than displaying stems such as 'danc'."""
    words = {}
    needed = set(terms)
    for doc_id in source_ids:
        lyrics = catalog.loc[doc_id, "lyrics"]
        tokens, spans = TextProcessor(lyrics).tokenize_with_spans()
        for term, (start, end) in zip(tokens, spans):
            if term in needed and term not in words:
                words[term] = lyrics[start:end].lower()
        if len(words) == len(needed):
            break
    return [words.get(term, term) for term in terms]


def prikazi_plejlistu(index, catalog, query: str, context: tuple,
                      ranked_ids: list[int], visible_ids: list[int]) -> None:
    """Once requested, refresh after every judgment or newly viewed results page."""
    displayed = st.session_state.setdefault("prikazane_pesme", {}).setdefault(context, set())
    displayed.update(visible_ids)
    context_key = sha256(repr(context).encode("utf-8")).hexdigest()
    active = st.session_state.setdefault("aktivne_plejliste", set())
    if st.button("Napravi plejlistu", key=f"napravi_plejlistu_{context_key}"):
        active.add(context)
    if context not in active:
        return

    judgments = st.session_state.get("ocene_pesama", {}).get(context, {})
    relevant = [doc_id for doc_id, choice in judgments.items() if choice == "like"]
    non_relevant = [doc_id for doc_id, choice in judgments.items() if choice == "dislike"]
    playlist = build_feedback_playlist(
        index, query, relevant, non_relevant, displayed_ids=displayed,
        initial_results=ranked_ids, families=catalog["porodica"].to_dict())
    # Keep the current result, not a history of stale playlists. The UI reads
    # fresh judgments every rerun, including removals and off-page selections.
    st.session_state.setdefault("plejliste", {})[context] = playlist

    st.divider()
    st.subheader("Tvoja plejlista")
    if playlist.assumed:
        st.info(f"Nema oznaka: kao pozitivni primeri korišćeno je prvih "
                f"{len(playlist.relevant)} prikazanih pesama.")
    if playlist.terms:
        words = _display_words([item.term for item in playlist.terms], playlist.relevant, catalog)
        st.text("Reči iz pesama koje ti se dopadaju: " + ", ".join(words))
    if playlist.avoided_terms:
        words = _display_words(playlist.avoided_terms, playlist.non_relevant, catalog)
        st.text("Umanjen uticaj reči iz pesama koje ti se ne dopadaju: " + ", ".join(words))
    if not playlist.songs:
        st.info("Nema drugih pesama za ovaj izbor. Probaj druge oznake ili upit.")
        return
    if len(playlist.songs) < 10:
        st.caption(f"Dostupno je {len(playlist.songs)} od 10 pesama nakon izostavljanja "
                   "već prikazanih pesama i njihovih verzija.")
    st.caption("Plejlista se automatski osvežava kada promeniš oznake.")
    for position, (doc_id, score) in enumerate(playlist.songs, 1):
        song = catalog.loc[doc_id]
        with st.container(border=True):
            st.text(f"{position}. {song['track_name']} — {song['track_artist']}")
            st.caption(f"Sličnost sa tvojim izborom: {score:.6f}")
