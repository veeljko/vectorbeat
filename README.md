# MrVectorbeat

Pocetni projekat za predmet Uvod u pronalazenje informacija. Tokom
semestra, korak po korak, pravite sopstveni muzicki pretrazivac nalik
Spotify-ju.

Algoritme pisete sami. Gotove biblioteke koristite samo tamo gde to kaze
specifikacija.

## Sta dobijate

Aplikacija se vec pokrece: bocni meni i sve strane postoje, ali su strane
prazne. Na svakoj strani u `app.py` stoji komentar `Koristi:` - koje
gotove funkcije tu mozete koristiti.

| Fajl | Sta je |
|---|---|
| `app.py` | sablon aplikacije - meni, prazne strane, funkcije za ucitavanje modela |
| `mrvectorbeat/pipeline.py` | sablon - prazne funkcije `napravi_X()` / `ucitaj_X()` za svaki model |
| `scripts/pripremi.py` | sablon - poziva sve `napravi_X()` redom, jednom, pre pokretanja aplikacije |
| `mrvectorbeat/prikaz.py` | GOTOV - izgled aplikacije (CSS, logo, kartice za rezultate) |
| `data/` | podaci (videti ispod) |
| `mrvectorbeat/__init__.py`, `scripts/__init__.py` | prazni fajlovi koji oznacavaju fasciklu kao Python paket - zbog njih radi `from mrvectorbeat import ...` i `python -m scripts.pripremi`. Ne brisati; svoje module stavljajte u `mrvectorbeat/` pored `pipeline.py`. |

Imena funkcija u sablonima, njihovi parametri i ono sto vracaju su samo
PREDLOG - menjajte ih, dodajte nove ili izbacite one koje vam ne trebaju.
Sve ostale module sa algoritmima (u `mrvectorbeat/`) pravite sami: kako ce se
zvati i sta ce biti u kom fajlu, odlucujete vi.

## Zahtevi

- Python 3.12 (proveravano na 3.12; moze i 3.13). Na starijim
  verzijama se biblioteke iz requirements.txt ne mogu instalirati.
- NVIDIA GPU nije obavezan, ali ubrzava rad sa neuronskim mrezama
- Ollama - za stranu AI DJ Bot
- Internet pri prvom pokretanju: neke biblioteke tada preuzimaju modele
  i recnike

## Instalacija

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Opciono, GPU (NVIDIA): torch iz requirements.txt je CPU verzija. CUDA verzija:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

Ollama (za AI DJ Bot-a): instalirati sa https://ollama.com, pa
```bash
ollama pull llama3.2:3b
```

PyCharm: kao interpreter projekta izaberite `.venv` (Add Interpreter ->
Add Local Interpreter -> Select existing -> `.venv\Scripts\python.exe`).
Inace PyCharm ne vidi instalirane biblioteke i podvlaci importe crveno.

## Podaci

| Fajl | Sta je | Koristi se |
|---|---|---|
| `data/spotify_songs.csv` | katalog pesama sa stihovima i audio-osobinama (Kaggle "Spotify Songs") | sve strane |
| `data/qrels.csv` | test upiti sa ocenama relevantnosti (`skup`, `upit`, `track_id`, `ocena`, `tipfeler`) | Evaluacija, Jezicki model, Hibridna pretraga |
| `data/music4all_istorija.csv` | istorija slusanja (`korisnik`, `track_id`, `vreme`), samo za pesme iz kataloga | Autoplay |

Graf izvodjaca i Wikipedia tekstove za AI DJ Bot-a pravi vas crawler.

Strana **Izvodjaci** prikazuje uticaj iz artefakta `artifacts/rangovi.pkl`.
PageRank se racuna nad celim prikupljenim grafom, ukljucujuci zanrove, albume
i izdavacke kuce; tek zatim se izdvajaju izvodjaci iz kataloga. Pored ocene
prikazuje se broj razlicitih stranica koje upucuju na izvodjaca.
Izabrano je `d = 0.85`: veze prenose najveci deo ranga, a nasumicni skokovi
ublazavaju zadrzavanje ranga u zatvorenim krugovima. Prikazane ocene nisu
ponovo normalizovane samo preko izvodjaca.

Uz vec pripremljen katalog, seedovi i graf pripremaju se komandom
`python -m scripts.pripremi --samo-graf-seed`, a uticaj komandom
`python -m scripts.pripremi --samo-rangovi`. Druga komanda ne pokrece crawler.
Puna priprema poziva oba koraka redom. Aplikacija samo ucitava gotov artefakt.

Na istoj strani odeljak **Slicni izvodjaci** koristi postojeci `graf.pkl` i
`artist_nodes.pkl`. Iz njih se izvodi indeks zajednickih ulaznih stranica;
stranica koja pokazuje na `k` razlicitih izvodjaca doprinosi skoru sa `1/k`.
Alternativna imena za isti cvor broje se jednom. Uz predloge se prikazuju
zajednicke stranice sa linkovima i njihovim doprinosima.
Indeks `artifacts/slicni_izvodjaci.pkl` priprema se komandom
`python -m scripts.pripremi --samo-slicni-izvodjaci`, ili u punoj pripremi.
Ovaj korak koristi vec sacuvan graf; aplikacija ocenjuje samo kandidate koji
dele ulaznu stranicu sa izabranim izvodjacem, bez novih HTTP zahteva.

## Pokretanje

Sve komande se pokrecu iz korena projekta, u terminalu u kom je
aktiviran `.venv` (`.venv\Scripts\activate`).

Prvo se grade modeli (jednom, i posle svake izmene koda koji ih gradi) -
sve se cuva u `artifacts/`:
```bash
python -m scripts.pripremi
```

Zatim aplikacija:
```bash
streamlit run app.py
```

Aplikacija se NE pokrece dugmetom Run u PyCharm-u (`python app.py`) -
samo komandom `streamlit run app.py`.

Ideja: modeli se grade JEDNOM (`napravi_X()` u `pripremi.py`), a
aplikacija ih samo ucitava (`ucitaj_X()` u `app.py`). Nista se ne gradi
iznova pri svakom kliku.

Vazno: u `.streamlit/config.toml` je `fileWatcherType = "none"` (inace
Streamlit u konzoli pada na transformers modulima), pa se aplikacija posle
SVAKE izmene koda mora rucno zaustaviti (Ctrl+C) i ponovo pokrenuti.

Katalog i Wikipedia tekstovi su na engleskom, pa i upiti treba da budu
na engleskom.
