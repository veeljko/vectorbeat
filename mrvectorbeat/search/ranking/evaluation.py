"""Priprema rang-lista i evaluacija iste liste porodica koju prikazuju Stihovi."""

import math

import pandas as pd

from mrvectorbeat.search.lyrics_search import search_lyrics
from .average_precision import average_precision
from .ndcg import ndcg_at_k
from .precision_at_k import precision_at_k
from .reciprocal_rank import reciprocal_rank


MODELS = {"tfidf": "TF-IDF", "cosine": "Kosinusna", "bm25": "BM25"}
GROUPS = {
    "poznata_bez_greske": "Jedna pesma bez greške",
    "poznata_sa_greskom": "Jedna pesma sa greškom",
    "tematski": "Tematski upiti",
}
K1_VALUES = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
B_VALUES = (0.0, 0.5, 1.0)


def prepare_queries(qrels: pd.DataFrame, catalog: pd.DataFrame) -> list[dict]:
    """Prenesi ocene track_id na porodice; sukob ocena resava veca ocena."""
    families_by_track = {}
    for row in catalog.itertuples(index=False):
        families_by_track.setdefault(row.track_id, set()).add(row.porodica)
    missing = set(qrels["track_id"]) - families_by_track.keys()
    if missing:
        raise ValueError(f"Qrels sadrzi {len(missing)} track_id vrednosti kojih nema u katalogu.")

    queries = []
    for (group, text, typo), rows in qrels.groupby(["skup", "upit", "tipfeler"], sort=False):
        grades = {}
        for row in rows.itertuples(index=False):
            for family in families_by_track[row.track_id]:
                grades[family] = max(grades.get(family, 0), int(row.ocena))
        queries.append({
            "query_id": f"q{len(queries) + 1:03d}", "upit": text,
            "skup": group, "tipfeler": int(typo),
            "group": ("tematski" if group == "tema" else
                      "poznata_sa_greskom" if typo else "poznata_bez_greske"),
            "grades": grades,
        })
    return queries


def metric_grades(query: dict) -> dict[str, int]:
    """Kod poznate stavke samo ocena 2 znaci relevantnost; teme cuvaju 2 i 1."""
    if query["skup"] == "poznata_stavka":
        return {family: 2 if grade == 2 else 0 for family, grade in query["grades"].items()}
    return query["grades"]


def query_metrics(query: dict, ranked_ids: list[int], songs: dict) -> dict[str, float]:
    """Oceni porodice u vec sacuvanom korisnickom poretku, bez novog rangiranja."""
    ranked = [songs[doc_id]["porodica"] for doc_id in ranked_ids]
    grades = metric_grades(query)
    return {
        "P@5": precision_at_k(ranked, grades, 5),
        "RR": reciprocal_rank(ranked, grades),
        "AP": average_precision(ranked, grades),
        "nDCG@5": ndcg_at_k(ranked, grades, 5),
    }


def group_tables(artifact: dict) -> dict[str, pd.DataFrame]:
    """Tri zasebna proseka, bez mesanja grupa i bez pristupa modelima."""
    tables = {}
    for group in GROUPS:
        queries = [q for q in artifact["queries"] if q["group"] == group]
        rows = []
        for model, label in MODELS.items():
            metrics = [query_metrics(q, q["rankings"][model], artifact["songs"]) for q in queries]
            means = {name: math.fsum(m[name] for m in metrics) / len(metrics) if metrics else 0.0
                     for name in ("P@5", "RR", "AP", "nDCG@5")}
            rows.append({"Model": label, "P@5": means["P@5"], "MRR": means["RR"],
                         "MAP": means["AP"], "nDCG@5": means["nDCG@5"]})
        tables[group] = pd.DataFrame(rows).set_index("Model")
    return tables


def build_evaluation(index, catalog: pd.DataFrame, qrels: pd.DataFrame, *,
                     k1_values=K1_VALUES, b_values=B_VALUES, progress=None) -> dict:
    """Samo priprema: pretrazi mrezu parametara i sacuvaj pune rang-liste tri modela."""
    if not k1_values or not b_values:
        raise ValueError("Mreza BM25 parametara ne sme biti prazna.")
    queries = prepare_queries(qrels, catalog)
    clean = [q for q in queries if q["group"] == "poznata_bez_greske"]
    if not clean:
        raise ValueError("Za izbor BM25 parametara potrebni su upiti za jednu pesmu bez greske.")
    families = catalog.set_index("katalog_id")["porodica"].to_dict()
    songs = {int(row.katalog_id): {"track_name": row.track_name, "track_artist": row.track_artist,
                                  "porodica": row.porodica}
             for row in catalog.itertuples(index=False)}
    grid = []
    for b in b_values:
        for k1 in k1_values:
            rrs = []
            for query in clean:
                result = search_lyrics(index, query["upit"], "ranked", measure="bm25",
                                       families=families, k1=k1, b=b, include_hits=False)
                ranked = [families[doc_id] for doc_id in result.scores]
                rrs.append(reciprocal_rank(ranked, metric_grades(query)))
            grid.append({"k1": float(k1), "b": float(b), "MRR": math.fsum(rrs) / len(rrs)})
            if progress:
                progress(f"BM25: k1={k1:g}, b={b:g}, MRR={grid[-1]['MRR']:.6f}")
    # Deterministicki izbor: najveci MRR, a kod izjednacenja manje b, pa manje k1.
    best = min(grid, key=lambda row: (-row["MRR"], row["b"], row["k1"]))
    selected = {"k1": best["k1"], "b": best["b"], "MRR": best["MRR"]}
    for number, query in enumerate(queries, start=1):
        query["rankings"] = {}
        for model in MODELS:
            result = search_lyrics(index, query["upit"], "ranked", measure=model,
                                   families=families, k1=selected["k1"], b=selected["b"],
                                   include_hits=False)
            query["rankings"][model] = list(result.scores)
        if progress and (number % 25 == 0 or number == len(queries)):
            progress(f"Evaluacija: sacuvane rang-liste za {number}/{len(queries)} upita")
    return {"version": 1, "queries": queries, "songs": songs,
            "bm25_grid": grid, "bm25_selected": selected}
