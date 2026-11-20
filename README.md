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
python -m nltk.downloader wordnet
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

`rangovi.pkl` cuva tabelu izvodjaca sa njihovim ocenama i brojem ulaznih veza,
uz parametre i metapodatke pripreme. Recnik PageRank ocena svih stranica koristi
se samo tokom racunanja i ne cuva se u artefaktu. Stariji format osvezi komandom
`python -m scripts.pripremi --samo-rangovi`.

Uz vec pripremljen katalog, seedovi i graf pripremaju se komandom
`python -m scripts.pripremi --samo-graf-seed`, a uticaj komandom
`python -m scripts.pripremi --samo-rangovi`. Druga komanda ne pokrece crawler.
Puna priprema poziva oba koraka redom. Aplikacija samo ucitava gotov artefakt.

Seedovi se prvo proveravaju grupnim Wikipedia API upitima za najvise 10 imena;
samo nereseni ili dvosmisleni slucajevi idu na dodatnu pretragu. Napredak se
cuva u `artifacts/seed.progress.parquet` posle svake grupe i pri prekidu rada.
Ponovno pokretanje nastavlja od zavrsenih imena, ukljucujuci proverena imena
bez rezultata. Zavrsni `seed.parquet` i dalje ima samo `artist_name` i `seed`,
a zamenjuje se tek kada se obradi ceo katalog.

HTML se preuzima serijski, sa najvise 10 stranica po izvodjacu. Svi HTTP
zahtevi, ukljucujuci seed API, robots.txt i svaki korak preusmeravanja, dele
isto zakljucavanje. Od zavrsetka jednog zahteva do pocetka sledeceg prolazi
najmanje jedna sekunda (`REQUEST_INTERVAL_SECONDS = 1.0` u crawler.py).
Trajna `requests.Session` ponovo koristi HTTP veze. Uspesno preuzete stranice
i `robots.txt` ostaju u `artifacts/crawler_pages/` i citaju se sa diska;
citanje iz kesa ne ceka pauzu za HTTP. Privremena opcija `--html-workers`
i paralelni obilazak su uklonjeni.

Kada su HTML stranice vec preuzete, graf se moze napraviti direktno iz kesa:

```bash
python -c "from mrvectorbeat.pipeline import napravi_graf; napravi_graf(cache_only=True)"
```

Ovaj rezim u prvom prolazu popisuje kanonske URL-ove svih sacuvanih Wikipedia
clanaka, a u drugom zadrzava samo veze izmedju njih. Nepreuzeti URL-ovi se ne
dodaju kao cvorovi; preusmeravanja se objedinjuju, a preuzete stranice bez veza
ostaju u grafu. Isto pravilo vazi i za zavrsni graf posle crawl-a.
Obradjuje se jedna HTML stranica odjednom, bez novog obilaska ili pretrage seedova.
Deljeni URL-ovi koriste isti objekat stringa da bi graf zauzimao manje memorije.
Posle gradnje pokreni `python -m scripts.pripremi --samo-rangovi`, pa
`python -m scripts.pripremi --samo-slicni-izvodjaci`.

Na istoj strani odeljak **Slicni izvodjaci** koristi postojeci `graf.pkl` i
`artist_nodes.pkl`. Iz njih se izvodi indeks zajednickih ulaznih stranica;
stranica koja pokazuje na `k` razlicitih izvodjaca doprinosi skoru sa `1/k`.
Alternativna imena za isti cvor broje se jednom. Uz predloge se prikazuju
zajednicke stranice sa linkovima i njihovim doprinosima.
Indeks `artifacts/slicni_izvodjaci.pkl` priprema se komandom
`python -m scripts.pripremi --samo-slicni-izvodjaci`, ili u punoj pripremi.
Ovaj korak koristi vec sacuvan graf; aplikacija ocenjuje samo kandidate koji
dele ulaznu stranicu sa izabranim izvodjacem, bez novih HTTP zahteva.

Strana **Jezicki model** koristi QLM sa Dirichlet ublazavanjem (`mu=1500`).
Ako nijedna od prvih pet pesama ne sadrzi bar 80% razlicitih obradjenih reci
upita, automatski se pokrece prosirenje sinonimima iz lokalnog WordNet recnika.
Ovo je pocetna heuristika pokrivenosti, a ne procena verovatnoce tacnog rezultata;
prag je `MIN_QUERY_COVERAGE` u `search/ranking/wordnet_expansion.py`.
WordNet se pretrazuje izvornim recima, pre stemovanja, uz najvise dva znacenja
po vrsti reci. Koriste se samo jednoclani sinonimi koji postoje u indeksu.
Postojeci Rocchio (`alpha=1`, `beta=0.75`, `gamma=0`) daje prednost sinonimima
podrzanim pocetnim rezultatima. Dodaje se najvise pet termina ukupno, po dva
za svaku rec upita, bez dupliranja stemova. Originalne reci zadrzavaju tezinu 1
po pojavljivanju, a sinonimi dobijaju 0.25 u zbiru QLM log-verovatnoca.
Posle jednog ponovnog rangiranja stvarno dodate reci ispisuju se iznad rezultata.
Prikaz sadrzanih/nedostajucih reci i dalje se odnosi samo na otkucani upit.
Ne priprema se novi indeks; WordNet se instalira gornjom komandom i zatim
koristi bez HTTP zahteva tokom pretrage. Ako recnik nedostaje, prikazuju se
pocetni rezultati sa uputstvom za instalaciju.

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
