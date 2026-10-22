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

# Katalog
katalog = pipeline.napravi_katalog()
print(f"Katalog: {len(katalog):,} pesama", flush=True)

# Tokeni
tokeni = pipeline.napravi_tokene()
print(f"Tokeni: {sum(map(len, tokeni['tokens'])):,} pojavljivanja", flush=True)

# Trie
pipeline.napravi_trie()

# K-gramski indeks
pipeline.napravi_kgram()

# Invertovani indeks
pipeline.napravi_indeks()

# TF-IDF model
pipeline.napravi_tfidf()

# BM25 i evaluacija
pipeline.napravi_bm25()
pipeline.napravi_evaluaciju()

# Jezicki model
pipeline.napravi_jezicki()

# PageRank nad grafom izvodjaca
# Vas crawler pravi graf izvodjaca - pokrenite ga pre ovog koraka.
pipeline.napravi_rangove()

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
