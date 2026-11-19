# scripts/pripremi.py
#
# SABLON. Skripta koja JEDNOM gradi sve modele i cuva ih u artifacts/,
# pre pokretanja aplikacije - app.py posle samo ucitava gotove fajlove.
#
# Pokretanje iz korena projekta:
#   python -m scripts.pripremi
#
# Poziva funkcije napravi_X() iz mrvectorbeat/pipeline.py. Dok su one
# prazne, skripta prolazi bez greske, ali nista ne gradi. Kad napisete
# neku funkciju, ispod njenog poziva dodajte ispis (sta je napravljeno,
# koliko cega) - tako odmah vidite da li radi.
#
# VAZNO: ovo nije fiksno - koraci i funkcije su samo PREDLOG, menjajte
# ih kako vam odgovara. Redosled je ipak bitan: kasniji koraci koriste
# ono sto su napravili raniji (indeks se gradi iz tokena, TF-IDF i BM25
# iz indeksa...).

from mrvectorbeat import pipeline


def _pripremi_graf_seed(*, retry_unresolved: bool = False):
    seedovi = pipeline.napravi_seed(retry_unresolved=retry_unresolved)
    print(f"Seedovi: {seedovi['seed'].notna().sum():,}/{len(seedovi):,} izvodjaca", flush=True)
    graf, artist_nodes = pipeline.napravi_graf()
    print(f"Graf: {len(graf):,} cvorova, {len(artist_nodes):,} izvodjaca", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Priprema modela i evaluacije.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--samo-evaluacija", action="store_true",
                       help="Koristi gotov katalog i indeks; pripremi rang-liste i BM25 graf.")
    modes.add_argument("--samo-graf-seed", action="store_true",
                       help="Koristi gotov katalog; pripremi seedove i graf, ponovi neresene seedove.")
    modes.add_argument("--samo-rangovi", action="store_true",
                       help="Koristi gotov katalog i graf; pripremi PageRank uticaj izvodjaca.")
    args = parser.parse_args()
    if args.samo_evaluacija:
        pipeline.napravi_evaluaciju()
        print("Evaluacija: gotovo", flush=True)
        return
    if args.samo_graf_seed:
        _pripremi_graf_seed(retry_unresolved=True)
        print("Seedovi i graf: gotovo", flush=True)
        return
    if args.samo_rangovi:
        rangovi = pipeline.napravi_rangove()
        print(f"PageRank: {len(rangovi['artists']):,} izvodjaca, d={rangovi['d']:g}", flush=True)
        return

    # Katalog
    katalog = pipeline.napravi_katalog()
    print(f"Katalog: {len(katalog):,} pesama", flush=True)

    # Tokeni
    tokeni = pipeline.napravi_tokene()
    print(f"Tokeni: {sum(map(len, tokeni['tokens'])):,} pojavljivanja", flush=True)

    # Trie
    pipeline.napravi_trie()

    # K-gramski indeks ostaje za kasnije.

    # Invertovani indeks
    indeks = pipeline.napravi_indeks()
    print(f"Pozicioni indeks: {len(indeks.index):,} termina", flush=True)

    # TF-IDF model
    pipeline.napravi_tfidf()

    # BM25 i evaluacija
    pipeline.napravi_bm25()
    pipeline.napravi_evaluaciju()

    # Jezicki model
    pipeline.napravi_jezicki()

    # Seedovi i graf: postojeci artefakti se ucitavaju, nedostajuci se pripremaju.
    _pripremi_graf_seed()

    # PageRank nad grafom izvodjaca
    rangovi = pipeline.napravi_rangove()
    print(f"PageRank: {len(rangovi['artists']):,} izvodjaca, d={rangovi['d']:g}", flush=True)

    # k-d stablo
    pipeline.napravi_kdrvo()

    # Semanticki model
    pipeline.napravi_semantiku()

    # RAG korpus
    pipeline.napravi_rag_korpus()

    # Hibridna pretraga
    pipeline.napravi_hibrid_evaluaciju()

    # Autoplay
    pipeline.napravi_autoplay()

    print("Priprema: gotovo")


if __name__ == "__main__":
    main()
