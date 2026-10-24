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

## Pretraga stihova

Strana **Stihovi** koristi jedan pozicioni indeks iz
`artifacts/pozicioni_indeks.pkl`. Gradi se iz sacuvanih tokena komandom
`python -m scripts.pripremi`, a aplikacija ga ucitava jednom i kesira.
Posle ponovne pripreme artefakata restartujte aplikaciju.

- **Bulova / AND**: pesma sadrzi sve termine upita.
- **Bulova / OR**: pesma sadrzi bar jedan termin.
- **Bulova / DIFFERENCE**: pesma sadrzi prvi termin, bez svih ostalih.
  Na primer, `love night rain` znaci `love` bez `night` i bez `rain`.
- **Frazna**: termini su uzastopni i istim redosledom.
- **Blizina**: posle obrade moraju ostati dva termina; trazi se
  `abs(pozicija_a - pozicija_b) < rastojanje`, u oba redosleda.
  Za rastojanje 2 prolaze susedni tokeni; za 1 nema pogodaka.

Upit prolazi istu obradu kao stihovi. Pozicije pocinju od nule i odnose
se na tokene posle uklanjanja stop-reci i stemovanja. Zato fraza
`love night` moze pogoditi originalni stih `love the night`. Za prikaz
se pozicije prevode u raspone originalnog teksta; rezultati prikazuju
do tri isecaka sa oznacenim pogocima, ukupan broj pesama i paginaciju.
Prazan obradjeni upit vraca prazan rezultat.

Struktura indeksa je `termin -> [(katalog_id, [pozicije]), ...]`.
Postings i pozicije su sortirane liste. Bulova pretraga iz iste
strukture izdvaja ID-eve, a frazna i blizina koriste i pozicije.
Presek, unija i razlika rade sa dva pokazivaca; presek krece od
najkrace liste i prekida obradu cim rezultat postane prazan.

Testovi se pokrecu iz korena projekta:

```bash
python -m unittest discover -s tests -v
```
