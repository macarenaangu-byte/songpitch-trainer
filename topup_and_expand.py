"""
topup_and_expand.py
─────────────────────────────────────────────────────────────────────
Phase 2 data collection for SongPitch genre model.

Goals:
  1. Top up all existing genres to 300 real originals
     (Country, R&B, Blues, Folk, Lo-Fi, Corporate, Cinematic, Gospel,
      Indie, Rock, Children's, Hip-Hop, EDM, Ambient, Afrobeats,
      Reggae, World Music — and diversify Latin + Jazz)
  2. Add 6 new genres from scratch (300 each):
     Funk_Soul, Trap, New_Age, Acoustic, House, Metal

Usage:
    python3 topup_and_expand.py
    # or in background:
    nohup /usr/bin/python3 topup_and_expand.py > topup2_log.txt 2>&1 &
"""

import os, time, requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_data')
TARGET_PER_GENRE = 300
MIN_FILE_BYTES   = 400_000   # ~25s at 128kbps
REQUEST_DELAY    = 0.25
MAX_WORKERS      = 8
print_lock       = Lock()


# ─── Genre queries ────────────────────────────────────────────────────────────
# Each genre lists multiple search queries.  The script stops as soon as the
# folder reaches TARGET_PER_GENRE files.  Genres already at target are skipped.

GENRE_QUERIES = {

    # ── Existing genres that need topping up ──────────────────────────────────

    "genre_Country": [
        "subject:country music nashville mediatype:audio",
        "subject:country folk americana mediatype:audio",
        "subject:bluegrass country mediatype:audio",
        "subject:country western music mediatype:audio",
        "subject:indie country alt-country mediatype:audio",
        "subject:country singer songwriter guitar mediatype:audio",
        "subject:country acoustic twang mediatype:audio",
    ],

    "genre_R&B": [
        "subject:r&b soul music mediatype:audio",
        "subject:rhythm blues contemporary mediatype:audio",
        "subject:neo soul r&b mediatype:audio",
        "subject:r&b smooth vocal mediatype:audio",
        "subject:contemporary rhythm blues mediatype:audio",
        "subject:r&b slow jam soul mediatype:audio",
        "subject:urban r&b vocal mediatype:audio",
    ],

    "genre_Corporate": [
        "subject:corporate background music mediatype:audio",
        "subject:business motivational background mediatype:audio",
        "subject:upbeat corporate positive music mediatype:audio",
        "subject:inspirational background corporate mediatype:audio",
        "subject:corporate ambient easy listening mediatype:audio",
        "subject:background music presentation mediatype:audio",
        "subject:uplifting business background music mediatype:audio",
    ],

    "genre_Cinematic": [
        "subject:cinematic trailer music mediatype:audio",
        "subject:epic orchestral dramatic mediatype:audio",
        "subject:cinematic dramatic tension mediatype:audio",
        "subject:epic trailer adventure cinematic mediatype:audio",
        "subject:cinematic score dramatic film mediatype:audio",
        "subject:dark cinematic orchestral tension mediatype:audio",
        "subject:action cinematic epic mediatype:audio",
    ],

    "genre_Folk": [
        "subject:folk acoustic singer songwriter mediatype:audio",
        "subject:folk guitar fingerpicking mediatype:audio",
        "subject:indie folk acoustic guitar mediatype:audio",
        "subject:folk traditional roots mediatype:audio",
        "subject:celtic folk acoustic mediatype:audio",
        "subject:american folk guitar vocals mediatype:audio",
        "subject:folk ballad acoustic guitar mediatype:audio",
    ],

    "genre_Lo_Fi": [
        "subject:lofi beats study chill mediatype:audio",
        "subject:lo-fi hip hop instrumental mediatype:audio",
        "subject:chillhop lofi beats mediatype:audio",
        "subject:lofi jazz piano beats mediatype:audio",
        "subject:lofi ambient study focus mediatype:audio",
        "subject:lo-fi bedroom music chill mediatype:audio",
        "subject:lofi vintage cassette beats mediatype:audio",
    ],

    "genre_Blues": [
        "subject:blues guitar electric mediatype:audio",
        "subject:chicago blues guitar mediatype:audio",
        "subject:delta blues acoustic guitar mediatype:audio",
        "subject:blues rock guitar slide mediatype:audio",
        "subject:blues harmonica guitar mediatype:audio",
        "subject:soul blues guitar vocal mediatype:audio",
        "subject:blues jazz guitar swing mediatype:audio",
    ],

    "genre_Gospel": [
        "subject:gospel choir music mediatype:audio",
        "subject:black gospel church choir mediatype:audio",
        "subject:southern gospel quartet mediatype:audio",
        "subject:gospel contemporary christian mediatype:audio",
        "subject:praise worship gospel piano mediatype:audio",
        "subject:gospel spiritual hymn mediatype:audio",
        "subject:gospel soul choir music mediatype:audio",
    ],

    "genre_Indie": [
        "subject:indie pop singer songwriter mediatype:audio",
        "subject:indie rock alternative guitar mediatype:audio",
        "subject:indie music bedroom pop lo-fi mediatype:audio",
        "subject:indie folk acoustic guitar vocal mediatype:audio",
        "subject:indie alternative college rock mediatype:audio",
        "subject:indie dream pop shoegaze mediatype:audio",
        "subject:indie singer songwriter piano mediatype:audio",
    ],

    "genre_Hip-Hop": [
        "subject:hip hop instrumental beats mediatype:audio",
        "subject:hip-hop rap underground mediatype:audio",
        "subject:boom bap hip hop beats mediatype:audio",
        "subject:hip hop old school beats mediatype:audio",
        "subject:hip hop rap vocal mediatype:audio",
        "subject:hip-hop conscious rap mediatype:audio",
        "subject:hip hop jazz beats sample mediatype:audio",
    ],

    "genre_EDM": [
        "subject:electronic dance music club mediatype:audio",
        "subject:edm dance trance music mediatype:audio",
        "subject:electronic rave dance beats mediatype:audio",
        "subject:trance electronic dance synth mediatype:audio",
        "subject:edm festival electronic music mediatype:audio",
        "subject:electronic dance pop synth mediatype:audio",
        "subject:edm dubstep electronic dance mediatype:audio",
    ],

    "genre_Ambient": [
        "subject:ambient drone soundscape mediatype:audio",
        "subject:ambient atmospheric electronic mediatype:audio",
        "subject:ambient space synthesizer mediatype:audio",
        "subject:ambient nature soundscape music mediatype:audio",
        "subject:dark ambient drone texture mediatype:audio",
        "subject:ambient cinematic drone pad mediatype:audio",
        "subject:ambient minimal electronic music mediatype:audio",
    ],

    "genre_Afrobeats": [
        "subject:afrobeats african pop music mediatype:audio",
        "subject:afrobeat nigerian highlife mediatype:audio",
        "subject:afropop west african music mediatype:audio",
        "subject:afrobeats dance music ghana nigeria mediatype:audio",
        "subject:afro fusion contemporary african mediatype:audio",
        "subject:afrobeats percussion vocal mediatype:audio",
        "subject:highlife afrobeat ghana music mediatype:audio",
    ],

    "genre_Reggae": [
        "subject:reggae jamaican music mediatype:audio",
        "subject:reggae dub roots music mediatype:audio",
        "subject:dancehall reggae jamaica mediatype:audio",
        "subject:roots reggae one drop mediatype:audio",
        "subject:reggae ska rocksteady mediatype:audio",
        "subject:dub reggae bass heavy mediatype:audio",
        "subject:reggae conscious roots rasta mediatype:audio",
    ],

    "genre_World_Music": [
        "subject:world music global fusion mediatype:audio",
        "subject:ethnic world music traditional mediatype:audio",
        "subject:global fusion cultural music mediatype:audio",
        "subject:middle eastern world music mediatype:audio",
        "subject:indian classical world music mediatype:audio",
        "subject:celtic world folk music mediatype:audio",
        "subject:world music percussion folk mediatype:audio",
    ],

    "genre_Rock": [
        "subject:rock guitar band music mediatype:audio",
        "subject:classic rock guitar riff mediatype:audio",
        "subject:hard rock guitar electric mediatype:audio",
        "subject:indie rock alternative guitar mediatype:audio",
        "subject:rock drums guitar bass mediatype:audio",
        "subject:psychedelic rock guitar mediatype:audio",
        "subject:rock power chord guitar band mediatype:audio",
    ],

    "genre_Childrens": [
        "subject:children songs fun educational mediatype:audio",
        "subject:nursery rhymes kids music mediatype:audio",
        "subject:children playful happy songs mediatype:audio",
        "subject:kids educational music piano mediatype:audio",
        "subject:children lullaby nursery songs mediatype:audio",
        "subject:kids songs playful silly mediatype:audio",
        "subject:children music sing-along fun mediatype:audio",
    ],

    # ── Latin: diversify sub-styles ───────────────────────────────────────────
    # Current 102 files skew Ibero-Latin. Add soukous, cumbia, bachata, bossa
    # nova, salsa so the model learns the full Latin/African-Latin spectrum.
    "genre_Latin": [
        "subject:cumbia latin music mediatype:audio",
        "subject:bachata merengue latin dance mediatype:audio",
        "subject:bossa nova brazil jazz mediatype:audio",
        "subject:salsa latin dance music mediatype:audio",
        "subject:soukous congolese rumba mediatype:audio",
        "subject:latin jazz afrobeat fusion mediatype:audio",
        "subject:samba latin brazil music mediatype:audio",
        "subject:latin pop dance rhythm mediatype:audio",
        "subject:flamenco latin guitar mediatype:audio",
    ],

    # ── Jazz: diversify sub-styles ────────────────────────────────────────────
    # Current 68 files are very narrow. Add bebop, fusion, smooth, swing.
    "genre_Jazz": [
        "subject:bebop jazz piano trumpet mediatype:audio",
        "subject:smooth jazz saxophone mediatype:audio",
        "subject:jazz fusion electric guitar mediatype:audio",
        "subject:swing jazz big band mediatype:audio",
        "subject:jazz piano trio bass drums mediatype:audio",
        "subject:contemporary jazz improvisation mediatype:audio",
        "subject:jazz blues saxophone guitar mediatype:audio",
        "subject:free jazz experimental mediatype:audio",
        "subject:jazz vocal standard mediatype:audio",
    ],

    # ── NEW genres (0 → 300) ──────────────────────────────────────────────────

    "genre_Funk_Soul": [
        "subject:funk soul groove music mediatype:audio",
        "subject:funk bass guitar groove mediatype:audio",
        "subject:soul r&b funk motown mediatype:audio",
        "subject:neo soul funk groove mediatype:audio",
        "subject:jazz funk groove bass mediatype:audio",
        "subject:classic soul funk rhythm mediatype:audio",
        "subject:funk pop groove dance mediatype:audio",
    ],

    "genre_Trap": [
        "subject:trap music hip hop 808 mediatype:audio",
        "subject:trap beats instrumental bass mediatype:audio",
        "subject:trap rap hip hop beats mediatype:audio",
        "subject:trap electronic hip hop bass mediatype:audio",
        "subject:trap music drill beats mediatype:audio",
        "subject:trap instrumental dark beats mediatype:audio",
        "subject:hip hop trap rnb bass 808 mediatype:audio",
    ],

    "genre_New_Age": [
        "subject:new age meditation ambient music mediatype:audio",
        "subject:spa relaxation healing music mediatype:audio",
        "subject:meditation yoga ambient music mediatype:audio",
        "subject:new age piano ambient healing mediatype:audio",
        "subject:spiritual ambient new age music mediatype:audio",
        "subject:nature sounds ambient relaxation mediatype:audio",
        "subject:new age crystal healing ambient mediatype:audio",
    ],

    "genre_Acoustic": [
        "subject:acoustic guitar singer songwriter mediatype:audio",
        "subject:acoustic fingerpicking guitar mediatype:audio",
        "subject:acoustic folk pop guitar vocal mediatype:audio",
        "subject:acoustic indie singer guitar mediatype:audio",
        "subject:acoustic guitar instrumental solo mediatype:audio",
        "subject:acoustic ballad guitar piano vocal mediatype:audio",
        "subject:acoustic live session guitar mediatype:audio",
    ],

    "genre_House": [
        "subject:house music deep electronic mediatype:audio",
        "subject:deep house chill electronic mediatype:audio",
        "subject:house club dance music 4x4 mediatype:audio",
        "subject:progressive house electronic dance mediatype:audio",
        "subject:house techno dance electronic mediatype:audio",
        "subject:deep house soulful vocals mediatype:audio",
        "subject:house music disco dance groove mediatype:audio",
    ],

    "genre_Metal": [
        "subject:heavy metal guitar distortion mediatype:audio",
        "subject:metal guitar riff heavy music mediatype:audio",
        "subject:doom metal heavy guitar mediatype:audio",
        "subject:progressive metal guitar technical mediatype:audio",
        "subject:thrash metal guitar fast heavy mediatype:audio",
        "subject:metal rock guitar drums heavy mediatype:audio",
        "subject:extreme metal heavy distortion guitar mediatype:audio",
    ],
}


# ─── Helpers (same pattern as collect_training_genres.py) ─────────────────────

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

    # Count only original files (not aug_ augmented ones) so we know real diversity
    existing_mp3s = {f for f in os.listdir(out_dir) if f.endswith(".mp3") and not f.startswith("aug_")}
    count = len(existing_mp3s)

    log(f"\n{'─'*56}")
    log(f"📂  {genre_key}  (have {count} originals, need {TARGET_PER_GENRE})")

    if count >= TARGET_PER_GENRE:
        log(f"   ✅ Already at target — skipping")
        return count

    needed = TARGET_PER_GENRE - count
    log(f"   🎯 Need {needed} more files")

    # Phase 1: collect download tasks (2× buffer)
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
                if safe not in existing_mp3s and not os.path.exists(dest):
                    tasks.append((ident, mp3name, dest))

    log(f"   📋 {len(tasks)} files queued for download")

    # Phase 2: parallel download
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

    final = len([f for f in os.listdir(out_dir) if f.endswith(".mp3") and not f.startswith("aug_")])
    status = "✅" if final >= TARGET_PER_GENRE else f"⚠️  only {final}"
    log(f"   {status}  Final originals: {final}  [{genre_key}]")
    return final


def main():
    print("=" * 56)
    print("  SongPitch Training Data — Phase 2 Top-Up + Expansion")
    print(f"  Target: {TARGET_PER_GENRE} original samples per genre")
    print(f"  Genres: {len(GENRE_QUERIES)}")
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
    print("  ✅ Data collection done.")
    print("  Next: bash run_training_pipeline.sh")


if __name__ == "__main__":
    main()
