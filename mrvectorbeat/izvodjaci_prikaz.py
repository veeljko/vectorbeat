"""Display artist influence and similarity from prepared artifacts."""

import re
import streamlit as st

from mrvectorbeat.search.ranking.similar_artists import find_similar_artists


def prikazi_uticaj(artifact: dict) -> None:
    st.caption("Uticaj se računa nad celim prikupljenim grafom Vikipedije. "
               "Rang-lista prikazuje samo izvođače iz kataloga.")
    for column, (label, value) in zip(st.columns(3), (
        ("Stranica u grafu", artifact["graph_nodes"]),
        ("Veza u grafu", artifact["graph_edges"]),
        ("Izvođača u rang-listi", len(artifact["artists"])),
    )):
        column.metric(label, value)

    d = artifact["d"]
    with st.expander("Kako se računa uticaj"):
        st.write(f"Faktor prigušenja d = {d:g}. Sa stranice koja ima izlazne veze, "
                 f"{d:.0%} ranga deli se tim vezama, dok se preostalih {1 - d:.0%} "
                 "ravnomerno raspodeljuje svim stranicama. Stranica bez izlaznih veza "
                 "raspodeljuje ceo svoj rang svim stranicama.")
        st.write("Podrazumevani izbor d = 0.85 daje veći značaj vezama među stranicama, uz redovne "
                 "skokove koji ublažavaju zadržavanje ranga u zatvorenim krugovima. "
                 "Izvođač zato dobija više uticaja kada na njega upućuju važne stranice.")
        st.write("Ulazne veze označavaju broj različitih stranica iz prikupljenog grafa "
                 "koje pokazuju na izvođača. Više linkova sa iste stranice broji se jednom. "
                 "PageRank ocena pripada ukupnom rangu svih stranica i nije "
                 "ponovo normalizovana samo preko izvođača.")

    if not artifact["converged"]:
        st.warning("PageRank je dostigao limit iteracija; prikazana je poslednja izračunata ocena.")
    missing = len(artifact["missing_artists"])
    if missing:
        st.caption(f"Za {missing} izvođača iz kataloga nema povezanog čvora u pripremljenom grafu.")
    table = artifact["artists"]
    if table.empty:
        st.info("U pripremljenom grafu još nema povezanih izvođača iz kataloga.")
        return
    query = st.text_input("Pretraži izvođača", key="uticaj_izvodjac").strip()
    if query:
        table = table.loc[table["artist_name"].str.contains(query, case=False, regex=False, na=False)]
    if table.empty:
        st.info("Nema izvođača za ovaj unos.")
        return
    st.dataframe(table.rename(columns={
        "position": "Pozicija", "artist_name": "Izvođač", "pagerank": "Uticaj (PageRank)",
        "incoming_links": "Ulazne veze", "url": "Vikipedija",
    })[["Pozicija", "Izvođač", "Uticaj (PageRank)", "Ulazne veze", "Vikipedija"]],
        hide_index=True, width="stretch", column_config={
            "Uticaj (PageRank)": st.column_config.NumberColumn(format="%.8f"),
            "Ulazne veze": st.column_config.NumberColumn(format="%d"),
            "Vikipedija": st.column_config.LinkColumn(display_text="Članak"),
        })


def _markdown_text(text: str) -> str:
    return re.sub(r"([\\`*_\[\]])", r"\\\1", text)


def prikazi_slicne_izvodjace(index: dict) -> None:
    st.caption("Izvođače povezuju stranice koje pokazuju na obojicu. "
               "Stranice koje povezuju manje izvođača daju veći doprinos sličnosti.")
    with st.expander("Kako se računa sličnost"):
        st.write("Zajednička stranica doprinosi sa 1/k, gde je k broj različitih izvođača "
                 "iz kataloga na koje pokazuje. Skor je zbir tih doprinosa. "
                 "Više imena za isti Vikipedijin čvor broji se kao jedan izvođač.")
    artists = sorted(index["artist_nodes"], key=lambda name: (name.casefold(), name))
    if not artists:
        st.info("U grafu još nema povezanih izvođača iz kataloga.")
        return
    artist = st.selectbox("Izaberite izvođača", artists, index=None,
                          placeholder="Pretražite izvođače iz kataloga", key="slicni_izvodjac")
    limit = st.selectbox("Broj sličnih izvođača", [5, 10, 20], index=1, key="slicni_izvodjaci_broj")
    if artist is None:
        st.info("Izaberite izvođača da biste videli srodne izvođače.")
        return
    results = find_similar_artists(index, artist, limit=limit)
    if not results:
        st.info("Nema drugih izvođača sa zajedničkim ulaznim stranicama u prikupljenom grafu.")
        return
    for position, result in enumerate(results, 1):
        st.markdown(f"**{position}. {_markdown_text(result['artist_name'])}**")
        st.caption(f"Sličnost: {result['score']:.6f} · Zajedničkih stranica: {len(result['shared_pages'])}")
        links = [f"[{_markdown_text(page['title'])}](<{page['url']}>)" for page in result["shared_pages"]]
        st.markdown("Oboje se pominju u: " + ", ".join(links))
        with st.expander(f"Doprinos zajedničkih stranica — {result['artist_name']}"):
            st.dataframe([{"Stranica": page["title"], "Doprinos": page["weight"]}
                          for page in result["shared_pages"]], hide_index=True, width="stretch",
                         column_config={"Doprinos": st.column_config.NumberColumn(format="%.6f")})
