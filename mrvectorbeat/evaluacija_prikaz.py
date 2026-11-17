"""Prikaz evaluacije iskljucivo iz pripremljenog artefakta."""

from html import escape

import pandas as pd
import streamlit as st

from mrvectorbeat.search.ranking.evaluation import GROUPS, MODELS, group_tables, query_metrics


def prikazi_evaluaciju(artifact: dict) -> None:
    st.caption("Rezultati koriste iste funkcije i jednu verziju svake porodice kao strana Stihovi. "
               "Upiti sa greškom se ne ispravljaju.")
    st.subheader("Tabele")
    tables = group_tables(artifact)
    for group, label in GROUPS.items():
        count = sum(q["group"] == group for q in artifact["queries"])
        st.markdown(f"**{label}** — {count} upita")
        main = "nDCG@5" if group == "tematski" else "MRR"
        st.caption(f"Glavna mera: {main}. Prikazane vrednosti su proseci samo za ovu grupu.")
        st.dataframe(tables[group], width="stretch", column_config={
            column: st.column_config.NumberColumn(format="%.4f")
            for column in ("P@5", "MRR", "MAP", "nDCG@5")
        })

    st.subheader("Pregled upita")
    st.caption("Zeleno: ocena 2 · Plavo: ocena 1 · Sivo: ocena 0. "
               "Za upit za jednu pesmu samo ocena 2 računa se kao pogodak.")
    queries = artifact["queries"]
    if queries:
        choice = st.selectbox("Upit iz fajla", range(len(queries)), format_func=lambda i:
                              f"{queries[i]['query_id']} · {GROUPS[queries[i]['group']]} · {queries[i]['upit']}")
        query = queries[choice]
        st.text(query["upit"])
        for column, (model, label) in zip(st.columns(3), MODELS.items()):
            with column:
                st.markdown(f"**{label}**")
                ranking = query["rankings"][model]
                if not ranking:
                    st.info("Nema rezultata.")
                for rank, doc_id in enumerate(ranking[:5], start=1):
                    song = artifact["songs"][doc_id]
                    grade = query["grades"].get(song["porodica"], 0)
                    color = {2: "#138a36", 1: "#2563eb", 0: "#737373"}[grade]
                    st.markdown(f'<p style="color: {color}">{rank}. '
                                f'{escape(str(song["track_name"]))} — {escape(str(song["track_artist"]))}</p>',
                                unsafe_allow_html=True)
                metrics = query_metrics(query, ranking, artifact["songs"])
                st.caption(" · ".join(f"{name}: {value:.4f}" for name, value in metrics.items()))

    st.subheader("Parametri BM25")
    grid = pd.DataFrame(artifact["bm25_grid"])
    chart = grid.pivot(index="k1", columns="b", values="MRR").rename(columns=lambda b: f"b={b:g}")
    st.line_chart(chart, x_label="k1", y_label="MRR")
    selected = artifact["bm25_selected"]
    st.info(f"Izabrano: k1 = {selected['k1']:g}, b = {selected['b']:g}; MRR = {selected['MRR']:.4f}. "
            "Ove početne vrednosti koriste se na strani Stihovi za BM25.")
    st.caption("Graf koristi samo upite za jednu pesmu bez greške. Izabran je najveći MRR u mreži; "
               "kod izjednačenja biraju se manje b, pa manje k1. "
               "Rang-liste i tačke grafa pripremljene su pre otvaranja ove strane.")
