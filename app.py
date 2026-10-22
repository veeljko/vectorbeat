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
    pass


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


ind = ucitaj_indeks()
porodice = ucitaj_porodice()
try:
    df = ucitaj_katalog()
except FileNotFoundError:
    df = None
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
        if query.count("*") > 1:
            pass
        elif "*" in query and not query.startswith("*") and not query.endswith("*"):
            # Pretraga sa zvezdicom u sredini ostaje za kasnije.
            pass
        elif query:
            suggestions = stablo.suggest(query, k=limit)
            if not suggestions:
                st.info("Nema predloga za ovaj unos.")
            for name in suggestions:
                key = name.casefold()
                # st.text cuva originalni naziv bez tumacenja HTML-a iz kataloga.
                st.text(name)
                st.caption(f"{', '.join(sorted(stablo.types[key]))} | "
                           f"Pojavljivanja: {stablo.frequencies[key]}")


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

    # Koristi: ind, ucitaj_tfidf(), porodice, df, prikaz.kartica()


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
