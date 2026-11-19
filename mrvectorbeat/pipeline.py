# mrvectorbeat/pipeline.py
#
# SABLON. Ovde su samo IMENA funkcija i opis sta svaka treba da radi -
# tela su prazna (pass), pa se svaka funkcija moze pozvati bez greske,
# ali jos nista ne radi.
#
# VAZNO: ovo NISU fiksne funkcije. Imena, parametri i ono sto funkcija
# vraca su samo PREDLOG - slobodno ih menjajte, dodajte nove ili
# izbacite one koje vam ne trebaju, kako vam odgovara.
#
# Ideja ovog fajla: za svaki model postoji par funkcija
#   napravi_X() - gradi model JEDNOM i cuva ga u artifacts/
#                 (poziva je scripts/pripremi.py)
#   ucitaj_X()  - cita vec sacuvan model iz artifacts/
#                 (poziva je app.py)
# Tako aplikacija nikad ne gradi nista iznova pri svakom kliku.

import re
import pickle
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from mrvectorbeat.search.name_search import NameSearch
from mrvectorbeat.search.indexes.position_index import PositionIndex


ROOT: Path = Path(__file__).resolve().parents[1]
ARTIFACTS: Path = ROOT / "artifacts"
CSV_PATH: Path = ROOT / "data" / "spotify_songs.csv"
KATALOG_PATH: Path = ARTIFACTS / "katalog.parquet"
TOKENI_PATH: Path = ARTIFACTS / "tokeni.parquet"
INDEKS_PATH: Path = ARTIFACTS / "pozicioni_indeks.pkl"
QRELS_PATH: Path = ROOT / "data" / "qrels.csv"
EVALUACIJA_PATH: Path = ARTIFACTS / "evaluacija.pkl"
SEED_PATH: Path = ARTIFACTS / "seed.parquet"
GRAF_PATH: Path = ARTIFACTS / "graf.pkl"
ARTIST_NODES_PATH: Path = ARTIFACTS / "artist_nodes.pkl"
RANGOVI_PATH: Path = ARTIFACTS / "rangovi.pkl"
SLICNI_IZVODJACI_PATH: Path = ARTIFACTS / "slicni_izvodjaci.pkl"


# Kolone koje zadrzavamo u katalogu. Sve ostalo iz CSV-a nam ne treba.
# Kartice u prikaz.py ocekuju bar: track_name, track_artist, godina,
# playlist_genre, tempo.
KOLONE: list[str] = [
    "katalog_id",         # indeks izvornog reda; cuva i ponovljene track_id
    "track_id",
    "track_name",
    "track_artist",
    "lyrics",
    "track_album_name",
    "godina",             # ovu kolonu pravimo mi
    "playlist_genre",
    "playlist_subgenre",
    "track_popularity",
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "porodica",           # ovu kolonu pravimo mi - isti kljuc za razlicite
                          # verzije iste pesme
]

_VERSION: re.Pattern[str] = re.compile(
    r"\b(?:remix(?:ed)?|remaster(?:ed)?|live|feat\.?|ft\.?|featuring)\b",
    re.IGNORECASE,
)
_BRACKETS: re.Pattern[str] = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")


def _normalizuj_ime(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(value).casefold()))


def kljuc_porodice(naslov: str, izvodjac: str) -> str:
    """Heuristika: ukloni oznake verzije, zadrzi ostale delove naslova."""
    #Izbacuje Remix, Live i slicno i ostavlja samo ime pesme
    #Song - 2004 Remaster → Song
    #Song - DJ Remix     → Song
    #Song - Live at Wembley → Song
    title: str = str(naslov)
    while True:
        cleaned: str = _BRACKETS.sub(
            lambda match: " " if _VERSION.search(match.group()) else match.group(), title
        )
        if cleaned == title:
            break
        title = cleaned
    # Izbacuje dodatak iza razdvojne crtice, npr. Song - 2004 Remaster.
    parts: list[str] = re.split(r"\s+[-\u2013\u2014]\s+", title)
    for index, part in enumerate(parts[1:], 1):
        if _VERSION.search(part):
            title = " - ".join(parts[:index])
            break
    title = re.split(r"\s+(?:feat\.?|ft\.?|featuring)\s+", title, flags=re.I)[0]
    title = re.sub(r"\s+(?:\d{4}\s+)?remaster(?:ed)?(?:\s+\d{4})?\s*$", "", title, flags=re.I)
    #Normalizuje imena
    #"  ARTIST! " → "artist"
    #"Song (Part Two)" → "song part two"
    base: str = _normalizuj_ime(title) or _normalizuj_ime(naslov)
    #Spaja artist::song kao identifikator
    return _normalizuj_ime(izvodjac) + "::" + base


def _snimi_tabelu(tabela: pd.DataFrame, putanja: Path) -> None:
    putanja.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = putanja.with_suffix(".tmp.parquet")
    tabela.to_parquet(temporary, index=False)
    temporary.replace(putanja)


def napravi_katalog() -> pd.DataFrame:
    """Procisti sirovi CSV, sacuvaj sve odgovarajuce redove i vrati DataFrame."""
    raw: pd.DataFrame = pd.read_csv(CSV_PATH, dtype={"track_id": "string"})

    #Boolean niz, True ako ta pesma ima language: en
    english: pd.Series = raw["language"].astype("string").str.strip().str.lower().eq("en")
    #Tekstovi pesama
    lyrics: pd.Series = raw["lyrics"].astype("string").str.strip()
    #Uzima samo podatke koji imaju language: en i cija pesma ima tekst
    catalog: pd.DataFrame = raw.loc[english.fillna(False) & lyrics.notna() & lyrics.ne("")].copy()

    catalog["lyrics"] = lyrics.loc[catalog.index]
    catalog["katalog_id"] = catalog.index

    #izvlaci datum i godinu
    dates: pd.Series = catalog["track_album_release_date"].astype("string").str.strip()
    years: pd.Series = dates.str.extract(r"^(\d{4})(?:-\d{2})?(?:-\d{2})?$", expand=False)
    catalog["godina"] = pd.to_numeric(years, errors="coerce").astype("Int64")

    #Ova petlja osigurava da naslov, izvođač i žanr postoje i imaju tekstualnu vrednost, kako bi kartice mogle da ih prikažu.
    for column in ["track_name", "track_artist", "playlist_genre"]:
        if column not in catalog:
            catalog[column] = pd.NA
        catalog[column] = catalog[column].fillna("Nepoznato").astype(str)
    
    catalog["porodica"] = [kljuc_porodice(name, artist) for name, artist in
                            zip(catalog["track_name"], catalog["track_artist"])]

    #Setuje odgovarajuci tip za svaku kolonu
    text_columns: set[str] = {"track_id", "track_name", "track_artist", "lyrics",
                    "track_album_name", "playlist_genre", "playlist_subgenre", "porodica"}
    for column in KOLONE:
        if column in text_columns:
            catalog[column] = catalog[column].astype("string")
        elif column not in {"katalog_id", "godina"}:
            catalog[column] = pd.to_numeric(catalog[column], errors="coerce").astype("Float64")

    catalog = catalog[KOLONE].reset_index(drop=True)
    _snimi_tabelu(catalog, KATALOG_PATH)
    return catalog


def ucitaj_katalog() -> pd.DataFrame:
    """Ucitaj iskljucivo gotov artefakt, bez pristupa sirovom CSV-u."""
    if not KATALOG_PATH.exists():
        raise FileNotFoundError("Katalog nije pripremljen. Pokreni: python -m scripts.pripremi")
    return pd.read_parquet(KATALOG_PATH)


def napravi_tokene() -> pd.DataFrame:
    """Vrati i sacuvaj tabelu katalog_id, track_id, tokens (lista stringova)."""
    # Lazy import: citanje kataloga u aplikaciji ne zahteva NLTK korpuse.
    from mrvectorbeat.textprocessing.text_processor import TextProcessor

    catalog: pd.DataFrame = ucitaj_katalog()
    tokens: pd.DataFrame = catalog[["katalog_id", "track_id"]].copy()
    # Sve instance koriste isti klasni kes; ne prazni se izmedju pesama.
    tokens["tokens"] = [TextProcessor(lyrics).text_tokenize(method="stem")
                        for lyrics in catalog["lyrics"]]
    _snimi_tabelu(tokens, TOKENI_PATH)
    print(tokens["tokens"])
    return tokens


def ucitaj_tokene() -> pd.DataFrame:
    """Ucitaj tabelu tokena iz artefakta; ne pokrece obradu teksta."""
    if not TOKENI_PATH.exists():
        raise FileNotFoundError("Tokeni nisu pripremljeni. Pokreni: python -m scripts.pripremi")
    tokens: pd.DataFrame = pd.read_parquet(TOKENI_PATH)
    tokens["tokens"] = tokens["tokens"].map(list)
    
    return tokens


def napravi_trie() -> None:
    """Gradi trie nad imenima pesama i izvodjaca i snima ga u artifacts/."""
    pass


def ucitaj_trie() -> NameSearch:
    """Ucitaj katalog iz artefakta i izgradi stabla za predloge imena.

    Aplikacija kesira rezultat pomocu st.cache_resource.
    """
    catalog: pd.DataFrame = ucitaj_katalog()
    names: Iterator[tuple[str, str]] = (
        (name, kind)
        for column, kind in [("track_name", "Pesma"), ("track_artist", "Izvodjac")]
        for name in catalog[column]
    )
    return NameSearch(names)


def ucitaj_porodice() -> None:
    """Vraca recnik track_id -> porodica."""
    pass


def ucitaj_popularnost() -> None:
    """Vraca recnik ime (pesma ili izvodjac) -> popularnost."""
    pass


def napravi_kgram() -> None:
    """Gradi k-gramski indeks nad imenima pesama i izvodjaca i snima ga u
    artifacts/."""
    pass


def ucitaj_kgram() -> None:
    """Cita gotov k-gramski indeks iz artifacts/."""
    pass


def napravi_indeks() -> PositionIndex:
    """Iz svih obradjenih stihova napravi jedan indeks za sve tipove upita."""
    catalog: pd.DataFrame = ucitaj_katalog().sort_values("katalog_id")
    tokens: pd.DataFrame = ucitaj_tokene().sort_values("katalog_id")
    if (catalog["katalog_id"].duplicated().any()
            or tokens["katalog_id"].duplicated().any()
            or catalog["katalog_id"].tolist() != tokens["katalog_id"].tolist()
            or catalog["track_id"].tolist() != tokens["track_id"].tolist()):
        raise ValueError("Katalog i tokeni nisu uskladjeni. Pokreni: python -m scripts.pripremi")
    index: PositionIndex = PositionIndex(
        tokens["tokens"].tolist(), [int(value) for value in tokens["katalog_id"]]
    )
    INDEKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = INDEKS_PATH.with_suffix(".tmp.pkl")
    with temporary.open("wb") as output:
        pickle.dump(index, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(INDEKS_PATH)
    return index


def ucitaj_indeks() -> PositionIndex:
    """Cita pripremljen indeks; aplikacija ga kesira pomocu cache_resource."""
    if not INDEKS_PATH.exists():
        raise FileNotFoundError("Indeks nije pripremljen. Pokreni: python -m scripts.pripremi")
    for source in (KATALOG_PATH, TOKENI_PATH):
        if source.exists() and source.stat().st_mtime_ns > INDEKS_PATH.stat().st_mtime_ns:
            raise ValueError("Indeks je zastareo. Pokreni: python -m scripts.pripremi")
    with INDEKS_PATH.open("rb") as source:
        index = pickle.load(source)
    if getattr(index, "ranking_version", None) != 2:
        raise ValueError("Indeks nema pripremljene tezine za rangiranje. Pokreni: python -m scripts.pripremi")
    return index


def napravi_tfidf() -> PositionIndex:
    """Tezine i norme vec su izgradjene i sacuvane u pozicionom indeksu."""
    return ucitaj_indeks()


def ucitaj_tfidf() -> PositionIndex:
    """Cita zajednicki artefakt za TF-IDF, kosinusnu i pozicionu pretragu."""
    return ucitaj_indeks()


def napravi_bm25() -> PositionIndex:
    """BM25 IDF i duzine vec su sacuvani pri gradnji pozicionog indeksa."""
    return ucitaj_indeks()


def ucitaj_bm25() -> PositionIndex:
    """Cita zajednicki artefakt sa pripremljenim BM25 statistikama."""
    return ucitaj_indeks()


def ucitaj_qrels(skup: str | None = None) -> pd.DataFrame:
    """Cita test upite iz data/qrels.csv.

    skup - "poznata_stavka", "tema", ili None za oba.
    Vraca: tabelu (skup, upit, track_id, ocena, tipfeler)."""
    qrels = pd.read_csv(QRELS_PATH, dtype={"track_id": "string", "upit": "string"})
    columns = ["skup", "upit", "track_id", "ocena", "tipfeler"]
    if not set(columns).issubset(qrels.columns) or qrels[columns].isna().any().any():
        raise ValueError("Qrels mora imati popunjene kolone skup, upit, track_id, ocena i tipfeler.")
    if (not qrels["skup"].isin(["poznata_stavka", "tema"]).all()
            or not qrels["ocena"].isin([0, 1, 2]).all()
            or not qrels["tipfeler"].isin([0, 1]).all()):
        raise ValueError("Qrels sadrzi nepoznatu grupu, ocenu ili oznaku tipfelera.")
    if skup is not None:
        if skup not in ("poznata_stavka", "tema"):
            raise ValueError("skup mora biti poznata_stavka ili tema.")
        qrels = qrels.loc[qrels["skup"] == skup]
    return qrels[columns].reset_index(drop=True)


def napravi_evaluaciju() -> dict:
    """Pripremi rang-liste tri modela i BM25 graf; aplikacija ih samo cita."""
    from mrvectorbeat.search.ranking.evaluation import build_evaluation

    artifact = build_evaluation(ucitaj_indeks(), ucitaj_katalog(), ucitaj_qrels(),
                                progress=lambda message: print(message, flush=True))
    EVALUACIJA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = EVALUACIJA_PATH.with_suffix(".tmp.pkl")
    with temporary.open("wb") as output:
        pickle.dump(artifact, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(EVALUACIJA_PATH)
    return artifact


def ucitaj_evaluaciju() -> dict:
    """Cita gotove rang-liste, ocene porodica i BM25 podatke, bez rangiranja."""
    if not EVALUACIJA_PATH.exists():
        raise FileNotFoundError("Evaluacija nije pripremljena. Pokreni: python -m scripts.pripremi --samo-evaluacija")
    for source in (QRELS_PATH, KATALOG_PATH, INDEKS_PATH):
        if source.exists() and source.stat().st_mtime_ns > EVALUACIJA_PATH.stat().st_mtime_ns:
            raise ValueError("Evaluacija je zastarela. Pokreni: python -m scripts.pripremi --samo-evaluacija")
    with EVALUACIJA_PATH.open("rb") as source:
        artifact = pickle.load(source)
    if artifact.get("version") != 1:
        raise ValueError("Format evaluacije nije podrzan. Ponovo pokreni scripts.pripremi.")
    return artifact


def napravi_jezicki(mu: float = 1000.0) -> None:
    """Gradi jezicki model i snima ga u artifacts/."""
    pass


def ucitaj_jezicki() -> None:
    """Cita gotov jezicki model iz artifacts/."""
    pass


def napravi_seed(*, retry_unresolved: bool = False) -> pd.DataFrame:
    """Ucitaj ili napravi seedove; opciono ponovi pretragu neresenih imena."""
    existing = None
    if SEED_PATH.exists():
        existing = ucitaj_seed()
        if not retry_unresolved or not (existing["seed"].isna() | existing["seed"].eq("")).any():
            return existing
    from mrvectorbeat.crawler.generate_seed import generate_seed

    return generate_seed(ucitaj_katalog(), SEED_PATH, existing=existing)


def ucitaj_seed() -> pd.DataFrame:
    """Cita samo pripremljene kolone artist_name i seed, bez HTTP zahteva."""
    if not SEED_PATH.exists():
        raise FileNotFoundError("Seedovi nisu pripremljeni. Pokreni: python -m scripts.pripremi")
    table = pd.read_parquet(SEED_PATH)
    if (list(table.columns) != ["artist_name", "seed"]
            or table["artist_name"].isna().any()
            or table["artist_name"].duplicated().any()):
        raise ValueError("Seed artefakt mora imati jedinstven artist_name i kolonu seed.")
    return table


def _snimi_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pkl")
    with temporary.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def napravi_graf(max_pages: int = 50, threshold: float = 2) -> tuple[dict, dict]:
    """Vrati (graf, artist_nodes); nedostajuce artefakte izgradi i sacuvaj.

    Prva gradnja obilazi svakog izvodjaca sa pronadjenim seedom i spaja veze u
    zajednicki graf. Sledeci pozivi citaju sacuvani graf i mapu izvodjaca.
    """
    seed_updated = SEED_PATH.stat().st_mtime_ns if SEED_PATH.exists() else 0
    graph_outdated = GRAF_PATH.exists() and seed_updated > GRAF_PATH.stat().st_mtime_ns
    nodes_outdated = ARTIST_NODES_PATH.exists() and seed_updated > ARTIST_NODES_PATH.stat().st_mtime_ns
    if (GRAF_PATH.exists() and ARTIST_NODES_PATH.exists()
            and not graph_outdated and not nodes_outdated):
        return ucitaj_graf()

    from mrvectorbeat.crawler.crawler import Crawler

    seeds = napravi_seed()
    artist_seeds = {artist: seed if isinstance(seed, str) and seed else None
                    for artist, seed in seeds.itertuples(index=False, name=None)}
    artists = list(artist_seeds)
    if GRAF_PATH.exists() and not graph_outdated:
        # A missing lookup can be rebuilt without downloading the graph again.
        with GRAF_PATH.open("rb") as source:
            graph = pickle.load(source)
    else:
        graph = {}
        if GRAF_PATH.exists():
            with GRAF_PATH.open("rb") as source:
                graph = pickle.load(source)
        crawler = Crawler()
        for artist, seed in artist_seeds.items():
            if seed is None:
                print(f"Preskacem izvodjaca bez potvrdjenog seeda: {artist}", flush=True)
                continue
            crawler.crawl_into_graph(artist, graph, max_pages=max_pages,
                                     threshold=threshold, seed=seed)
        _snimi_pickle(graph, GRAF_PATH)
    artist_nodes = Crawler.build_artist_nodes(artists, graph, artist_seeds=artist_seeds)
    _snimi_pickle(artist_nodes, ARTIST_NODES_PATH)
    return graph, artist_nodes


def ucitaj_graf() -> tuple[dict, dict]:
    """Ucitaj pripremljeni graf i mapu izvodjaca, bez gradnje ili HTTP zahteva."""
    if not GRAF_PATH.exists():
        raise FileNotFoundError("Graf nije pripremljen. Pokreni: python -m scripts.pripremi")
    if not ARTIST_NODES_PATH.exists():
        raise FileNotFoundError("Mapa izvodjaca nije pripremljena. Pokreni: python -m scripts.pripremi")
    with GRAF_PATH.open("rb") as source:
        graph = pickle.load(source)
    with ARTIST_NODES_PATH.open("rb") as source:
        artist_nodes = pickle.load(source)
    return graph, artist_nodes


def napravi_rangove(d: float = 0.85) -> dict:
    """Pripremi uticaj nad celim grafom i rang-listu izvodjaca iz kataloga."""
    if RANGOVI_PATH.exists():
        try:
            artifact = ucitaj_rangove()
        except ValueError:
            pass
        else:
            if artifact["d"] == d:
                return artifact
    from mrvectorbeat.search.ranking.artist_influence import build_artist_influence

    graph, artist_nodes = ucitaj_graf()
    if SEED_PATH.exists() and any(
        SEED_PATH.stat().st_mtime_ns > source.stat().st_mtime_ns
        for source in (GRAF_PATH, ARTIST_NODES_PATH)
    ):
        raise ValueError("Graf izvodjaca je zastareo u odnosu na seedove. "
                         "Pokreni: python -m scripts.pripremi --samo-graf-seed, "
                         "zatim python -m scripts.pripremi --samo-rangovi")
    catalog = ucitaj_katalog()
    artifact = build_artist_influence(graph, artist_nodes, catalog["track_artist"], d=d)
    _snimi_pickle(artifact, RANGOVI_PATH)
    return artifact


def ucitaj_rangove() -> dict:
    """Cita pripremljene ocene i ulazne veze; ne racuna PageRank ni crawling."""
    command = "Pokreni: python -m scripts.pripremi --samo-rangovi"
    if not RANGOVI_PATH.exists():
        raise FileNotFoundError("Uticaj izvodjaca nije pripremljen. " + command)
    for source in (GRAF_PATH, ARTIST_NODES_PATH, KATALOG_PATH, SEED_PATH):
        if source.exists() and source.stat().st_mtime_ns > RANGOVI_PATH.stat().st_mtime_ns:
            raise ValueError("Rangovi izvodjaca su zastareli. " + command)
    with RANGOVI_PATH.open("rb") as source:
        artifact = pickle.load(source)
    if not isinstance(artifact, dict) or artifact.get("version") != 1:
        raise ValueError("Format rangova izvodjaca nije podrzan. " + command)
    return artifact


def napravi_slicne_izvodjace() -> dict:
    """Iz postojecih graf i artist_nodes artefakata izvedi indeks slicnosti."""
    if SLICNI_IZVODJACI_PATH.exists():
        try:
            return ucitaj_slicne_izvodjace()
        except ValueError:
            pass
    from mrvectorbeat.search.ranking.similar_artists import build_similarity_index

    graph, artist_nodes = ucitaj_graf()
    if SEED_PATH.exists() and any(
        SEED_PATH.stat().st_mtime_ns > source.stat().st_mtime_ns
        for source in (GRAF_PATH, ARTIST_NODES_PATH)
    ):
        raise ValueError("Graf izvodjaca je zastareo u odnosu na seedove. "
                         "Pokreni: python -m scripts.pripremi --samo-graf-seed, "
                         "zatim python -m scripts.pripremi --samo-slicni-izvodjaci")
    artifact = build_similarity_index(graph, artist_nodes)
    _snimi_pickle(artifact, SLICNI_IZVODJACI_PATH)
    return artifact


def ucitaj_slicne_izvodjace() -> dict:
    """Ucitaj pripremljene ulazne veze i tezine, bez obrade grafa ili HTTP-a."""
    command = "Pokreni: python -m scripts.pripremi --samo-slicni-izvodjaci"
    if not SLICNI_IZVODJACI_PATH.exists():
        raise FileNotFoundError("Indeks slicnih izvodjaca nije pripremljen. " + command)
    for source in (GRAF_PATH, ARTIST_NODES_PATH, SEED_PATH):
        if source.exists() and source.stat().st_mtime_ns > SLICNI_IZVODJACI_PATH.stat().st_mtime_ns:
            raise ValueError("Indeks slicnih izvodjaca je zastareo. " + command)
    with SLICNI_IZVODJACI_PATH.open("rb") as source:
        artifact = pickle.load(source)
    if not isinstance(artifact, dict) or artifact.get("version") != 1:
        raise ValueError("Format indeksa slicnih izvodjaca nije podrzan. " + command)
    return artifact


def napravi_kdrvo() -> None:
    """Gradi k-d stablo nad audio-osobinama kataloga i snima ga u
    artifacts/."""
    pass


def ucitaj_kdrvo() -> None:
    """Cita gotovo k-d stablo iz artifacts/."""
    pass


def napravi_semantiku() -> None:
    """Gradi semanticki model nad katalogom i snima ga u artifacts/."""
    pass


def ucitaj_semantiku() -> None:
    """Cita gotov semanticki model iz artifacts/."""
    pass


def napravi_rag_korpus() -> None:
    """Gradi korpus za AI DJ Bot-a i snima ga u artifacts/."""
    pass


def ucitaj_rag_korpus() -> None:
    """Cita gotov RAG korpus iz artifacts/."""
    pass


def napravi_hibrid_evaluaciju() -> None:
    """Racuna metrike hibridne pretrage nad qrels i snima tabelu u
    artifacts/."""
    pass


def ucitaj_hibrid_evaluaciju() -> None:
    """Cita gotovu tabelu metrika hibridne pretrage iz artifacts/."""
    pass


def napravi_autoplay() -> None:
    """Gradi Autoplay nad data/music4all_istorija.csv i snima ga u
    artifacts/."""
    pass


def ucitaj_autoplay() -> None:
    """Cita gotov Autoplay iz artifacts/."""
    pass
