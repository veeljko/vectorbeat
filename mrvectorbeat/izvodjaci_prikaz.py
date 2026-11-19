"""Display artist influence exclusively from the prepared ranking artifact."""

import streamlit as st


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
