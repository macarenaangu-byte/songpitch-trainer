"""
phase3_expand.py
─────────────────────────────────────────────────────────────────────
Phase 3 genre expansion for SongPitch.

22 new genres → brings total model from 31 to 53 genres.

  Latin:      Bachata, Cumbia, Merengue, Tango, Flamenco,
              Trap Latino, Reggaetón, Dancehall
  Electronic: Techno, Trance, Drum & Bass, Dubstep, Synthwave
  Rock/Alt:   Punk, Hard Rock, Alternative Rock, Grunge, Progressive Rock
  Classical:  Opera, Baroque
  Urban:      HyperPop, Urbano

Usage:
    python3 phase3_expand.py
    # or background:
    nohup /usr/bin/python3 phase3_expand.py > phase3_log.txt 2>&1 &
"""

import os, time, requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_data')
TARGET_PER_GENRE = 300
MIN_FILE_BYTES   = 400_000
REQUEST_DELAY    = 0.25
MAX_WORKERS      = 16
print_lock       = Lock()

GENRE_QUERIES = {

    # ── Latin sub-genres ──────────────────────────────────────────────────────

    "genre_Bachata": [
        "subject:bachata music romance mediatype:audio",
        "subject:bachata dominicana guitar vocal mediatype:audio",
        "subject:bachata latin dance music mediatype:audio",
        "subject:bachata romantic latin guitar mediatype:audio",
        "subject:bachata sensual dance latin mediatype:audio",
        "subject:bachata moderna guitar vocal rhythm mediatype:audio",
        "subject:latin bachata dance caribbean mediatype:audio",
    ],

    "genre_Cumbia": [
        "subject:cumbia colombiana music mediatype:audio",
        "subject:cumbia latin dance accordion mediatype:audio",
        "subject:cumbia tropical rhythm mediatype:audio",
        "subject:cumbia peru argentina music mediatype:audio",
        "subject:cumbia folk dance latin mediatype:audio",
        "subject:cumbia band accordion percussion mediatype:audio",
        "subject:cumbia sonidera dance music mediatype:audio",
    ],

    "genre_Merengue": [
        "subject:merengue dominicano music mediatype:audio",
        "subject:merengue latin dance mediatype:audio",
        "subject:merengue accordion percussion rhythm mediatype:audio",
        "subject:merengue caribbean dance music mediatype:audio",
        "subject:merengue ritmo latino dance mediatype:audio",
        "subject:merengue dominican republic music mediatype:audio",
        "subject:merengue fast latin dance rhythm mediatype:audio",
    ],

    "genre_Tango": [
        "subject:tango argentino music mediatype:audio",
        "subject:tango bandoneon piano orchestra mediatype:audio",
        "subject:tango dance argentina music mediatype:audio",
        "subject:tango piazzolla nuevo mediatype:audio",
        "subject:tango milonga dance music mediatype:audio",
        "subject:tango violin bandoneon classical mediatype:audio",
        "subject:tango orchestra instrumental mediatype:audio",
    ],

    "genre_Flamenco": [
        "subject:flamenco guitar music spain mediatype:audio",
        "subject:flamenco spanish guitar vocal mediatype:audio",
        "subject:flamenco dance music andalusian mediatype:audio",
        "subject:flamenco cajon guitar vocal mediatype:audio",
        "subject:flamenco puro guitar traditional mediatype:audio",
        "subject:flamenco classical spanish guitar mediatype:audio",
        "subject:flamenco cante jondo guitar music mediatype:audio",
    ],

    "genre_Trap_Latino": [
        "subject:trap latino urban latin mediatype:audio",
        "subject:latin trap urban spanish beats mediatype:audio",
        "subject:trap rap latin urban spanish mediatype:audio",
        "subject:latin trap music beats bass mediatype:audio",
        "subject:trap urbano latin hip hop mediatype:audio",
        "subject:latin trap drill beats urban mediatype:audio",
        "subject:trap urban latin contemporary mediatype:audio",
    ],

    "genre_Reggaeton": [
        "subject:reggaeton music urban latino mediatype:audio",
        "subject:reggaeton dembow rhythm latin mediatype:audio",
        "subject:reggaeton latin urban contemporary mediatype:audio",
        "subject:reggaeton instrumental beats dembow mediatype:audio",
        "subject:reggaeton trap urban latino music mediatype:audio",
        "subject:reggaeton dance latin urban music mediatype:audio",
        "subject:reggaeton perreo urban latin beats mediatype:audio",
    ],

    "genre_Dancehall": [
        "subject:dancehall reggae jamaican music mediatype:audio",
        "subject:dancehall riddim jamaican beats mediatype:audio",
        "subject:dancehall jamaican dance music mediatype:audio",
        "subject:dancehall deejay music jamaica mediatype:audio",
        "subject:dancehall bashment caribbean music mediatype:audio",
        "subject:dancehall reggae soca caribbean mediatype:audio",
        "subject:dancehall riddim rhythm jamaican mediatype:audio",
    ],

    # ── Electronic sub-genres ─────────────────────────────────────────────────

    "genre_Techno": [
        "subject:techno electronic music dark mediatype:audio",
        "subject:techno berlin minimal electronic mediatype:audio",
        "subject:techno industrial electronic beats mediatype:audio",
        "subject:techno minimal synthesizer dance mediatype:audio",
        "subject:techno club electronic music mediatype:audio",
        "subject:techno hard electronic industrial mediatype:audio",
        "subject:techno dark rave electronic music mediatype:audio",
    ],

    "genre_Trance": [
        "subject:trance electronic music uplifting mediatype:audio",
        "subject:progressive trance electronic synth mediatype:audio",
        "subject:trance synth electronic anthem mediatype:audio",
        "subject:psytrance psychedelic electronic music mediatype:audio",
        "subject:trance eurodance electronic music mediatype:audio",
        "subject:trance uplifting synth electronic mediatype:audio",
        "subject:vocal trance electronic anthem mediatype:audio",
    ],

    "genre_Drum_and_Bass": [
        "subject:drum and bass electronic music mediatype:audio",
        "subject:drum bass dnb electronic breakbeat mediatype:audio",
        "subject:jungle drum bass breaks electronic mediatype:audio",
        "subject:liquid drum bass electronic music mediatype:audio",
        "subject:neurofunk drum bass electronic mediatype:audio",
        "subject:drum bass rave electronic music mediatype:audio",
        "subject:drum bass breakbeat jungle electronic mediatype:audio",
    ],

    "genre_Dubstep": [
        "subject:dubstep electronic music bass mediatype:audio",
        "subject:dubstep wobble bass electronic mediatype:audio",
        "subject:dubstep uk electronic bass music mediatype:audio",
        "subject:dubstep electronic dance bass drop mediatype:audio",
        "subject:dubstep brostep heavy bass electronic mediatype:audio",
        "subject:dubstep garage bass music electronic mediatype:audio",
        "subject:dubstep bass heavy electronic music mediatype:audio",
    ],

    "genre_Synthwave": [
        "subject:synthwave retro synth electronic mediatype:audio",
        "subject:synthwave 80s retro electronic music mediatype:audio",
        "subject:retrowave synthwave electronic synth mediatype:audio",
        "subject:outrun synthwave retro electronic mediatype:audio",
        "subject:synthwave dark retro neon electronic mediatype:audio",
        "subject:synthwave electronic retro 80s music mediatype:audio",
        "subject:darksynth synthwave electronic retro mediatype:audio",
    ],

    # ── Rock / Alt sub-genres ─────────────────────────────────────────────────

    "genre_Punk": [
        "subject:punk rock music guitar fast mediatype:audio",
        "subject:punk music guitar drums energy mediatype:audio",
        "subject:punk hardcore fast guitar music mediatype:audio",
        "subject:punk pop rock guitar music mediatype:audio",
        "subject:classic punk rock guitar vocals mediatype:audio",
        "subject:punk rock rebellion guitar band mediatype:audio",
        "subject:punk oi rock guitar fast mediatype:audio",
    ],

    "genre_Hard_Rock": [
        "subject:hard rock heavy guitar electric mediatype:audio",
        "subject:hard rock guitar riff drums band mediatype:audio",
        "subject:hard rock classic guitar power mediatype:audio",
        "subject:hard rock arena guitar band mediatype:audio",
        "subject:heavy rock guitar distortion drums mediatype:audio",
        "subject:hard rock guitar chord classic mediatype:audio",
        "subject:hard rock electric guitar band music mediatype:audio",
    ],

    "genre_Alternative_Rock": [
        "subject:alternative rock guitar music mediatype:audio",
        "subject:alternative rock 90s guitar band mediatype:audio",
        "subject:alt rock guitar indie alternative mediatype:audio",
        "subject:alternative rock post-grunge guitar mediatype:audio",
        "subject:alternative music guitar band vocals mediatype:audio",
        "subject:alternative rock college indie guitar mediatype:audio",
        "subject:alternative rock guitar distortion music mediatype:audio",
    ],

    "genre_Grunge": [
        "subject:grunge rock guitar heavy music mediatype:audio",
        "subject:grunge distortion guitar rock music mediatype:audio",
        "subject:grunge alternative heavy guitar music mediatype:audio",
        "subject:grunge 90s rock guitar distortion mediatype:audio",
        "subject:grunge alt rock guitar heavy music mediatype:audio",
        "subject:grunge seattle rock guitar music mediatype:audio",
        "subject:grunge heavy guitar band rock music mediatype:audio",
    ],

    "genre_Progressive_Rock": [
        "subject:progressive rock complex guitar keyboard mediatype:audio",
        "subject:prog rock keyboard guitar art music mediatype:audio",
        "subject:progressive rock art rock guitar mediatype:audio",
        "subject:progressive rock synth keyboard guitar mediatype:audio",
        "subject:prog rock experimental guitar complex mediatype:audio",
        "subject:progressive rock long form guitar keyboard mediatype:audio",
        "subject:art rock progressive experimental guitar mediatype:audio",
    ],

    # ── Classical sub-genres ──────────────────────────────────────────────────

    "genre_Opera": [
        "subject:opera classical vocal music mediatype:audio",
        "subject:opera aria classical soprano tenor mediatype:audio",
        "subject:classical opera voice orchestra mediatype:audio",
        "subject:opera italian classical voice music mediatype:audio",
        "subject:opera orchestra vocal soprano mediatype:audio",
        "subject:opera singer aria orchestral classical mediatype:audio",
        "subject:opera classical dramatic vocal music mediatype:audio",
    ],

    "genre_Baroque": [
        "subject:baroque classical music harpsichord mediatype:audio",
        "subject:baroque harpsichord orchestra classical mediatype:audio",
        "subject:baroque Bach Handel Vivaldi music mediatype:audio",
        "subject:baroque period classical ensemble mediatype:audio",
        "subject:baroque chamber music early music mediatype:audio",
        "subject:baroque continuo early classical music mediatype:audio",
        "subject:baroque orchestral suite classical mediatype:audio",
    ],

    # ── Urban ─────────────────────────────────────────────────────────────────

    "genre_HyperPop": [
        "subject:hyperpop electronic music mediatype:audio",
        "subject:hyperpop pc music electronic pop mediatype:audio",
        "subject:hyperpop experimental pop electronic mediatype:audio",
        "subject:hyperpop bubblegum bass electronic mediatype:audio",
        "subject:hyperpop digicore electronic music mediatype:audio",
        "subject:hyperpop glitchy electronic pop music mediatype:audio",
        "subject:hyperpop experimental electronic music pop mediatype:audio",
    ],

    "genre_Urbano": [
        "subject:urbano latino contemporary music mediatype:audio",
        "subject:latin urban contemporary music mediatype:audio",
        "subject:urbano latin trap reggaeton mediatype:audio",
        "subject:latin urban pop contemporary beats mediatype:audio",
        "subject:urbano contemporary latin music mediatype:audio",
        "subject:latin urban afrobeats fusion mediatype:audio",
        "subject:urbano latin contemporary pop beats mediatype:audio",
    ],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg):
    with print_lock:
        print(msg, flush=True)


def search_archive(query, rows=80):
    try:
        r = requests.get("https://archive.org/advancedsearch.php", params={
            "q": f"({query}) AND mediatype:audio AND format:MP3",
            "fl[]": ["identifier", "title"],
            "rows": rows, "output": "json", "page": 1,
        }, timeout=15)
        return r.json().get("response", {}).get("docs", []) if r.status_code == 200 else []
    except Exception as e:
        log(f"  ⚠️  Search error: {e}")
        return []


def get_mp3s(ident):
    try:
        r = requests.get(f"https://archive.org/metadata/{ident}", timeout=10)
        if r.status_code == 200:
            return [f["name"] for f in r.json().get("files", [])
                    if f.get("name", "").lower().endswith(".mp3")]
    except Exception:
        pass
    return []


def download_one(ident, mp3name, dest_path):
    url = f"https://archive.org/download/{ident}/{quote(mp3name)}"
    try:
        r = requests.get(url, timeout=60, stream=True, allow_redirects=True)
        if r.status_code == 200:
            cl = int(r.headers.get("content-length", 0))
            if 0 < cl < MIN_FILE_BYTES:
                return None
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            if os.path.getsize(dest_path) >= MIN_FILE_BYTES:
                return dest_path
            os.remove(dest_path)
    except Exception:
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
    return None


def collect_genre(genre_key, queries):
    out_dir = os.path.join(OUTPUT_DIR, genre_key)
    os.makedirs(out_dir, exist_ok=True)

    existing = {f for f in os.listdir(out_dir)
                if f.endswith(".mp3") and not f.startswith("aug_")}
    count = len(existing)

    log(f"\n{'─'*56}")
    log(f"📂  {genre_key}  (have {count} originals, need {TARGET_PER_GENRE})")

    if count >= TARGET_PER_GENRE:
        log(f"   ✅ Already at target — skipping")
        return count

    tasks = []
    seen_idents = set()

    for query in queries:
        if len(tasks) + count >= TARGET_PER_GENRE * 3:
            break
        items = search_archive(query, rows=80)
        time.sleep(REQUEST_DELAY)
        for item in items:
            ident = item.get("identifier", "")
            if not ident or ident in seen_idents:
                continue
            seen_idents.add(ident)
            mp3s = get_mp3s(ident)
            time.sleep(REQUEST_DELAY)
            for mp3name in mp3s:
                safe = f"arch_{ident[:28]}_{mp3name[:38].replace(' ','_').replace('/','_')}"
                dest = os.path.join(out_dir, safe)
                if safe not in existing and not os.path.exists(dest):
                    tasks.append((ident, mp3name, dest))

    log(f"   📋 {len(tasks)} files queued for download")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, t[0], t[1], t[2]): t[2] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                count += 1
                if count % 25 == 0 or count == TARGET_PER_GENRE:
                    log(f"   ✅  {count}/{TARGET_PER_GENRE} collected — {genre_key}")
            if count >= TARGET_PER_GENRE:
                for f in futures:
                    f.cancel()
                break

    final = len([f for f in os.listdir(out_dir)
                 if f.endswith(".mp3") and not f.startswith("aug_")])
    status = "✅" if final >= TARGET_PER_GENRE else f"⚠️  only {final}"
    log(f"   {status}  Final originals: {final}  [{genre_key}]")
    return final


def main():
    print("=" * 56)
    print("  SongPitch Training Data — Phase 3 Expansion")
    print(f"  Target: {TARGET_PER_GENRE} original samples per genre")
    print(f"  New genres: {len(GENRE_QUERIES)}")
    print("  Total after this run: 31 existing + 22 new = 53 genres")
    print("=" * 56)

    totals = {}
    for genre_key, queries in GENRE_QUERIES.items():
        totals[genre_key] = collect_genre(genre_key, queries)

    print("\n" + "=" * 56)
    print("  FINAL SUMMARY")
    print("=" * 56)
    for genre_key, count in totals.items():
        status = "✅" if count >= TARGET_PER_GENRE else f"⚠️  {count}/{TARGET_PER_GENRE}"
        print(f"  {status}  {genre_key}")

    print()
    print("  ✅ Phase 3 collection done.")
    print("  Next: bash run_training_pipeline.sh")


if __name__ == "__main__":
    main()
