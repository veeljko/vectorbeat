# app.py
#
# SABLON aplikacije. Bocna traka i sve strane vec postoje, ali su strane
# prazne - na svakoj je samo naslov i komentar koje gotove funkcije tu
# mozete koristiti.
#
# Funkcije za ucitavanje (ucitaj_X ispod) pozivaju pipeline.ucitaj_X().
# Dok su one u pipeline.py prazne, vracaju None - aplikacija se pokrece
# bez greske, samo jos nema sta da prikaze.
#
# VAZNO: ovo nije fiksno - imena funkcija, sta primaju i sta vracaju su
# samo PREDLOG, menjajte ih kako vam odgovara.
#
# Pokretanje iz korena projekta:
#   streamlit run app.py

import streamlit as st
from st_keyup import st_keyup

from mrvectorbeat import pipeline, prikaz

# OVDE IDU importi vasih modula, npr.:
#   from mrvectorbeat import tekst, upit, indeks, tfidf, bm25, ...

st.set_page_config(page_title="Mr Vectorbeat", layout="wide")
st.markdown(prikaz.CSS, unsafe_allow_html=True)


# --- Ucitavanje modela ---------------------------------------------------
#
# Svaka funkcija je obavijena dekoratorom da se ucita SAMO JEDNOM, a ne
# pri svakom kliku:
#   @st.cache_data      - za tabele i recnike (pandas DataFrame, dict)
#   @st.cache_resource  - za modele (objekti klasa, FAISS indeks, ...)

@st.cache_resource
def ucitaj_indeks():
    return pipeline.ucitaj_indeks()


@st.cache_resource
def ucitaj_trie():
    return pipeline.ucitaj_trie()


@st.cache_data
def ucitaj_katalog():
    return pipeline.ucitaj_katalog()


@st.cache_data
def ucitaj_izvodjace():
    pass


@st.cache_data
def broj_reci():
    pass


@st.cache_resource
def ucitaj_tfidf():
    return pipeline.ucitaj_tfidf()


@st.cache_resource
def ucitaj_bm25():
    return pipeline.ucitaj_bm25()


@st.cache_data
def ucitaj_evaluaciju():
    return pipeline.ucitaj_evaluaciju()


@st.cache_data
def ucitaj_qrels():
    return pipeline.ucitaj_qrels()


@st.cache_data
def ucitaj_graf():
    return pipeline.ucitaj_graf()


@st.cache_data
def ucitaj_porodice():
    return pipeline.ucitaj_porodice()


@st.cache_data
def ucitaj_popularnost():
    return pipeline.ucitaj_popularnost()


@st.cache_resource
def ucitaj_kdrvo():
    return pipeline.ucitaj_kdrvo()


@st.cache_resource
def ucitaj_semantiku():
    return pipeline.ucitaj_semantiku()


@st.cache_resource
def ucitaj_rag_korpus():
    return pipeline.ucitaj_rag_korpus()


@st.cache_resource
def ucitaj_autoplay():
    return pipeline.ucitaj_autoplay()


@st.cache_data
def ucitaj_hibrid_evaluaciju():
    return pipeline.ucitaj_hibrid_evaluaciju()


@st.cache_resource
def ucitaj_sbert():
    pass


@st.cache_resource
def zagrej_ollama():
    pass


@st.cache_data
def ucitaj_labele_pesama():
    pass


@st.cache_data
def ucitaj_labele_autoplay():
    pass


@st.cache_data
def ucitaj_katalog_po_id():
    return ucitaj_katalog().set_index("katalog_id", drop=False)


@st.cache_data(max_entries=200)
def napravi_isecke(lyrics, hits):
    from mrvectorbeat.search.lyrics_search import make_snippets
    return make_snippets(lyrics, hits)


@st.cache_resource
def ucitaj_jezicki(mu):
    pass


@st.cache_data
def ucitaj_tokene_po_pesmi():
    pass


@st.cache_data
def ucitaj_audio_vektore():
    pass


@st.cache_data
def ucitaj_sbert_vektore():
    pass


porodice = ucitaj_porodice()
try:
    df = ucitaj_katalog()
except FileNotFoundError:
    df = None
try:
    ind = ucitaj_indeks() if df is not None else None
    index_error = ""
except (FileNotFoundError, ValueError) as error:
    ind = None
    index_error = str(error)
stablo = ucitaj_trie() if df is not None else None
izvodjaci = ucitaj_izvodjace()
# SBERT i Ollama se ucitavaju odmah pri pokretanju - trebaju i "Po smislu"
# i "AI DJ Bot", pa cekanje ide na start, a ne na prvi upit.
ucitaj_sbert()
zagrej_ollama()


# --- Bocna traka -------------------------------------------------------

with st.sidebar:
    st.markdown(prikaz.LOGO, unsafe_allow_html=True)

    strana = st.radio("Izaberite stranu", ["Obrada teksta", "Pretraga", "Stihovi", "Jezicki model", "Evaluacija",
                                           "Izvodjaci", "Katalog", "Radio", "Po smislu", "AI DJ Bot",
                                           "Hibridna pretraga", "Klasteri", "Autoplay"])

    st.divider()
    st.caption("Ucitani indeksi")
    # OVDE IDE statistika preko prikaz.statistika(labela, broj):
    #   prikaz.statistika("Pesama", len(df))
    #   prikaz.statistika("Izvodjaca", len(izvodjaci))
    #   prikaz.statistika("Reci u indeksu", broj_reci())


# --- Pretraga ----------------------------------------------------------

if strana == "Pretraga":
    st.title("Pretraga")
    st.caption("Kucajte naziv pesme ili izvodjaca.")

    if stablo is None:
        st.info("Prvo pripremi katalog: python -m scripts.pripremi")
    else:
        query = st_keyup("Ime pesme ili izvodjaca", debounce=200, key="pretraga_imena") or ""
        limit = st.selectbox("Broj predloga", [5, 10, 20], index=1)
        if query:
            suggestions = stablo.suggest(query, k=limit)
            if not suggestions:
                st.info("Nema predloga za ovaj unos.")
            if suggestions and "*" not in query and not suggestions[0].casefold().startswith(query.casefold()):
                st.info("Nema dopune za ovaj unos. Da li ste mislili:")
            for name in suggestions:
                # st.text cuva originalni naziv bez tumacenja HTML-a iz kataloga.
                st.text(name)
                st.caption(f"Pojavljivanja: {stablo.forward.get_frequency(name)}")


# --- Katalog -----------------------------------------------------------

elif strana == "Katalog":
    st.title("Katalog")

    if df is None:
        st.info("Katalog jos nije pripremljen.")
        st.code("python -m scripts.pripremi", language="bash")
    else:
        controls = st.columns(4)
        genre = controls[0].selectbox("Zanr", ["Svi zanrovi"] + sorted(df["playlist_genre"].unique()))
        sort_label = controls[1].selectbox("Sortiraj po", ["Popularnost", "Godina izdanja", "Tempo"])
        direction = controls[2].selectbox("Redosled", ["Opadajuce", "Rastuce"])
        page_size = controls[3].selectbox("Pesama po strani", [12, 24, 48], index=1)
        selected = df if genre == "Svi zanrovi" else df[df["playlist_genre"] == genre]
        sort_column = {"Popularnost": "track_popularity", "Godina izdanja": "godina", "Tempo": "tempo"}[sort_label]
        selected = selected.sort_values(sort_column, ascending=direction == "Rastuce",
                                        na_position="last", kind="stable")
        total = len(selected)
        pages = max(1, (total + page_size - 1) // page_size)
        selection = (genre, sort_label, direction, page_size)
        if st.session_state.get("katalog_izbor") != selection:
            st.session_state["katalog_strana"] = 1
            st.session_state["katalog_izbor"] = selection
        page = st.number_input("Strana", min_value=1, max_value=pages, step=1,
                               key="katalog_strana")
        st.caption(f"Ukupno pesama: {total:,} | Strana {page} od {pages}")
        if total == 0:
            st.info("Nema pesama za izabrane filtere.")
        start = (page - 1) * page_size
        visible = selected.iloc[start:start + page_size]
        for offset in range(0, len(visible), 4):
            columns = st.columns(4)
            for column, (_, row) in zip(columns, visible.iloc[offset:offset + 4].iterrows()):
                with column:
                    card = row.copy()
                    card["godina"] = "Nepoznato" if row["godina"] is None or str(row["godina"]) == "<NA>" else str(int(row["godina"]))
                    prikaz.plocica(card)


# --- Stihovi -----------------------------------------------------------

elif strana == "Stihovi":
    st.title("Pretraga po stihovima")
    st.caption("Pretrazuje sadrzaj pesama, ne naslove.")

    if ind is None:
        st.info(index_error or "Prvo pripremite katalog i indeks: python -m scripts.pripremi")
    else:
        from mrvectorbeat.search.lyrics_search import search_lyrics

        mode_label = st.radio("Rezim pretrage", ["Bulova", "Frazna", "Blizina"], horizontal=True)
        mode = {"Bulova": "boolean", "Frazna": "phrase", "Blizina": "proximity"}[mode_label]
        operator = "AND"
        distance = 3
        if mode == "boolean":
            operator = st.selectbox("Operator", ["AND", "OR", "DIFFERENCE"])
            st.caption({"AND": "Pesma mora sadrzati sve unete termine.",
                        "OR": "Pesma mora sadrzati bar jedan uneti termin.",
                        "DIFFERENCE": "Pesma sadrzi prvi termin, a ne sadrzi nijedan od ostalih."}[operator])
        elif mode == "proximity":
            distance = st.number_input("Rastojanje manje od", min_value=1, value=3, step=1)
            st.caption("Unesite dva termina. Redosled nije bitan; susedni tokeni imaju rastojanje 1.")
        st.caption("Upit i stihovi se obradjuju isto: mala slova, tokenizacija, stop-reci, stemovanje. "
                   "Fraza i rastojanje proveravaju se nad tako obradjenim tokenima.")
        query = st.text_input("Unesite deo stihova", key="stihovi_upit")
        if query.strip():
            try:
                results = search_lyrics(ind, query, mode, operator, distance)
            except ValueError as error:
                st.info(str(error))
            else:
                if not results.terms:
                    st.info("Posle obrade upita nije ostao nijedan termin.")
                elif not results.hits:
                    st.info("Nema pesama koje odgovaraju upitu.")
                else:
                    st.caption("Obradjeni termini: " + ", ".join(results.terms))
                    page_size = st.selectbox("Rezultata po strani", [10, 20, 50])
                    total = len(results.hits)
                    pages = (total + page_size - 1) // page_size
                    selection = (query, mode, operator, distance, page_size)
                    if st.session_state.get("stihovi_izbor") != selection:
                        st.session_state["stihovi_strana"] = 1
                        st.session_state["stihovi_izbor"] = selection
                    page = st.number_input("Strana rezultata", min_value=1, max_value=pages,
                                           step=1, key="stihovi_strana")
                    st.caption(f"Ukupno pesama: {total:,} | Strana {page} od {pages}")
                    catalog_by_id = ucitaj_katalog_po_id()
                    start = (page - 1) * page_size
                    for doc_id in list(results.hits)[start:start + page_size]:
                        song = catalog_by_id.loc[doc_id]
                        with st.container(border=True):
                            st.text(f"{song['track_name']} — {song['track_artist']}")
                            st.caption(f"Godina: {song['godina']} | Zanr: {song['playlist_genre']}")
                            for snippet in napravi_isecke(song["lyrics"], results.hits[doc_id]):
                                st.markdown('<div style="white-space: pre-wrap">' + snippet + '</div>',
                                            unsafe_allow_html=True)


# --- Jezicki model -----------------------------------------------------

elif strana == "Jezicki model":
    st.title("Jezicki model i prosirenje upita")
    st.caption("Treci nacin rangiranja, i sta se sve moze dodati upitu.")

    # Koristi: ucitaj_jezicki(mu), ucitaj_bm25(), ucitaj_tokene_po_pesmi(),
    #          ucitaj_qrels(), ind, df, prikaz.kartica()


# --- Evaluacija --------------------------------------------------------

elif strana == "Evaluacija":
    st.title("Evaluacija pretrage")

    # Koristi: ucitaj_evaluaciju(), ucitaj_qrels(), ucitaj_tfidf(),
    #          ucitaj_bm25(), ind, df, prikaz.kartica()


# --- Izvodjaci ---------------------------------------------------------

elif strana == "Izvodjaci":
    st.title("Uticaj izvodjaca")
    st.caption("Ko je najuticajniji izvodjac - PageRank nad grafom koji je "
               "napravio crawler, naspram prostog brojanja linkova.")

    # Koristi: ucitaj_graf(), prikaz.kartica_izvodjaca()


# --- Radio -------------------------------------------------------------

elif strana == "Radio":
    st.title("Radio — slicne pesme po zvuku")
    st.caption("Pretraga po audio-osobinama (tempo, energija, glasnoca...), "
               "ne po tekstu ni zanru. k-d stablo + k-NN nad 9 osobina.")

    # Koristi: ucitaj_kdrvo(), ucitaj_labele_pesama(), ucitaj_katalog_po_id(),
    #          porodice, prikaz.kartica_radio()


# --- Po smislu ---------------------------------------------------------

elif strana == "Po smislu":
    st.title("Pretraga po smislu")
    st.caption("Opisi raspolozenje obicnim jezikom - sistem razume smisao, "
               "ne samo tacne reci iz stihova.")

    # Koristi: ucitaj_semantiku(), ucitaj_sbert(), ucitaj_katalog_po_id(),
    #          prikaz.kartica_smisao()


# --- AI DJ Bot ---------------------------------------------------------

elif strana == "AI DJ Bot":
    st.title("AI DJ Bot")
    st.caption("Pitaj o izvodjacima - bot odgovara na osnovu Wikipedia "
               "clanaka i pokazuje odakle je uzeo odgovor.")

    # Koristi: ucitaj_rag_korpus(), ucitaj_sbert()


# --- Klasteri ----------------------------------------------------------

elif strana == "Klasteri":
    st.title("Klasteri — grupisanje bez unapred poznatih oznaka")
    st.caption("K-means nad dva razlicita prostora: zvuk (9 audio-osobina) "
               "naspram smisla teksta (SBERT).")

    # Koristi: ucitaj_audio_vektore(), ucitaj_sbert_vektore(), df


# --- Autoplay ----------------------------------------------------------

elif strana == "Autoplay":
    st.title("Autoplay — predvidjanje sledece pesme")
    st.caption("Markovljev lanac naspram Transformera, oba obucena na "
               "stvarnoj istoriji slusanja (data/music4all_istorija.csv).")

    # Koristi: ucitaj_autoplay(), ucitaj_labele_autoplay(),
    #          ucitaj_katalog_po_id(), prikaz.kartica()


# --- Hibridna pretraga -------------------------------------------------

elif strana == "Hibridna pretraga":
    st.title("Hibridna pretraga — BM25 + SBERT (RRF)")
    st.caption("Isti upit kroz tacne reci (BM25) i kroz smisao (SBERT), pa "
               "spojeno preko Reciprocal Rank Fusion.")

    # Koristi: ucitaj_bm25(), ucitaj_semantiku(), ind, porodice,
    #          ucitaj_hibrid_evaluaciju(), prikaz.kartica()


# --- Obrada teksta -----------------------------------------------------

else:
    st.title("Obrada teksta")
    st.caption("Kako se stihovi pretvaraju u reci kojima pretrazujemo.")

    # Koristi: vas modul za obradu teksta
