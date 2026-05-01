"""
collect_training_genres.py
Downloads training audio for new genre categories from Archive.org (Jamendo CC tracks).
Uses parallel downloads for speed. Target: 300 samples per new genre.
Usage: python3 collect_training_genres.py
"""

import os, time, sys, requests
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'training_data')
TARGET_PER_GENRE   = 300
MIN_FILE_BYTES     = 400_000   # ~25s at 128kbps
REQUEST_DELAY      = 0.25
MAX_WORKERS        = 8
print_lock         = Lock()

GENRE_QUERIES = {
    "genre_Classical": [
        "subject:classical piano",
        "subject:classical violin strings",
        "subject:classical orchestra chamber",
        "subject:bach beethoven mozart mediatype:audio",
        "subject:classical chopin schubert brahms",
        "subject:contemporary classical neoclassical",
        "subject:classical piano solo",
    ],
    "genre_Musical_Theatre": [
        "subject:musical theater mediatype:audio",
        "subject:broadway musical mediatype:audio",
        "subject:musical theatre songs mediatype:audio",
        "subject:show tunes broadway mediatype:audio",
        "subject:musical cast recording mediatype:audio",
        "subject:showtune mediatype:audio",
        "subject:musical comedy theater songs mediatype:audio",
    ],
    "genre_Gospel": [
        "subject:gospel choir mediatype:audio",
        "subject:gospel music hymn mediatype:audio",
        "subject:gospel spiritual christian mediatype:audio",
        "subject:contemporary christian gospel mediatype:audio",
        "subject:praise worship gospel mediatype:audio",
        "subject:church choir gospel mediatype:audio",
        "subject:sacred music gospel hymns mediatype:audio",
    ],
    "genre_Lo_Fi": [
        "subject:lofi mediatype:audio",
        "subject:lo-fi hip hop chill mediatype:audio",
        "subject:chillhop beats mediatype:audio",
        "subject:lo-fi jazz beats mediatype:audio",
        "subject:lofi study music mediatype:audio",
        "subject:ambient lofi chill mediatype:audio",
        "subject:lofi piano beats mediatype:audio",
    ],
    "genre_Corporate": [
        "subject:corporate background music mediatype:audio",
        "subject:business background music mediatype:audio",
        "subject:corporate ambient motivational mediatype:audio",
        "subject:easy listening background music mediatype:audio",
        "subject:corporate uplifting positive mediatype:audio",
        "subject:background music upbeat corporate mediatype:audio",
        "subject:business presentation background mediatype:audio",
    ],
    "genre_Cinematic": [
        "subject:cinematic trailer music mediatype:audio",
        "subject:epic cinematic orchestral mediatype:audio",
        "subject:trailer music epic mediatype:audio",
        "subject:cinematic dramatic film score mediatype:audio",
        "subject:epic orchestral trailer dramatic mediatype:audio",
        "subject:cinematic adventure action mediatype:audio",
        "subject:dark cinematic tension mediatype:audio",
    ],
    "genre_Childrens": [
        "subject:children music songs mediatype:audio",
        "subject:nursery rhymes children mediatype:audio",
        "subject:kids music playful mediatype:audio",
        "subject:children songs piano mediatype:audio",
        "subject:educational children music mediatype:audio",
        "subject:kids songs happy playful mediatype:audio",
        "subject:children lullaby nursery mediatype:audio",
    ],
    "genre_KPop": [
        "subject:kpop mediatype:audio",
        "subject:k-pop korean pop mediatype:audio",
        "subject:korean pop music mediatype:audio",
        "subject:kpop instrumental mediatype:audio",
        "subject:korean indie pop synth mediatype:audio",
        "subject:k-pop style beats mediatype:audio",
        "subject:kpop dance pop mediatype:audio",
    ],
}


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
            return [f["name"] for f in r.json().get("files", []) if f.get("name", "").lower().endswith(".mp3")]
    except Exception:
        pass
    return []


def download_one(ident, mp3name, dest_path):
    """Download a single file. Returns dest_path on success, None on failure."""
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
            try: os.remove(dest_path)
            except: pass
    return None


def collect_genre(genre_key, queries):
    out_dir = os.path.join(OUTPUT_DIR, genre_key)
    os.makedirs(out_dir, exist_ok=True)
    existing = {f for f in os.listdir(out_dir) if f.endswith(".mp3")}
    count = len(existing)

    log(f"\n{'─'*56}")
    log(f"📂  {genre_key}  (have {count}, need {TARGET_PER_GENRE})")

    if count >= TARGET_PER_GENRE:
        log(f"   ✅ Already at target")
        return count

    # Phase 1: collect all (ident, mp3name, dest) download tasks
    tasks = []
    seen_idents = set()

    for query in queries:
        if len(tasks) + count >= TARGET_PER_GENRE * 2:  # collect 2× so parallell has buffer
            break
        items = search_archive(query, rows=60)
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
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    final = len([f for f in os.listdir(out_dir) if f.endswith(".mp3")])
    status = "✅" if final >= TARGET_PER_GENRE else f"⚠️  only {final}"
    log(f"   {status}  Final count: {final}  [{genre_key}]")
    return final


def main():
    print("=" * 56)
    print("  SongPitch Training Data Collector")
    print(f"  Target: {TARGET_PER_GENRE} samples per genre × {len(GENRE_QUERIES)} genres")
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
    print("  Next: python3 final_boss_train.py")


if __name__ == "__main__":
    main()
