#!/usr/bin/env python3
"""
label_tool.py — Local music genre labeling tool for YAMNet training data.

USAGE:
  python3 label_tool.py <source_folder> [output_folder]

  source_folder  folder of unlabeled MP3/WAV/etc files to listen through
  output_folder  where genre subfolders are created  (default: ./labeled_audio)

Then open http://localhost:8765 in your browser and start labeling.
  Space      play / pause
  →          skip (unsure)
  ⌘Z / CtrlZ undo last label
  Enter      confirm selected genres

You can assign up to 3 genres per track. Click genres to select/deselect,
then click "Label & Next" (or press Enter) to confirm and advance.

WHERE TO FIND FREE TRACKS (no copyright issues):
  • Jamendo        jamendo.com          — search by genre, free CC download
  • YouTube Audio  studio.youtube.com   — Music Library → filter by genre
  • Free Music Archive  freemusicarchive.org  — browse by genre, CC licensed

AFTER LABELING:
  python3 ingest_labeled_audio.py <output_folder>   # extract embeddings
  python3 train_yamnet_classifier.py                # retrain model
  git add yamnet_genre_model.h5 yamnet_genre_model_encoder.pkl
  git commit -m "Retrain genre model with hand-labeled data"
  # review with your team before pushing — every push deploys to production
"""

import os, sys, json, shutil, mimetypes, urllib.parse, webbrowser, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

SOURCE_DIR = None
OUTPUT_DIR = None
AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.aiff', '.m4a'}
PORT       = 8765
_history   = []   # (primary_dst, original_src, [secondary_dsts])
_presort   = {}   # filename → {prediction, confidence, top3}

# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Genre Label Tool</title>
<style>
:root {
  --bg:          #0F1117;
  --surface:     #161B26;
  --surface2:    #1D2235;
  --border:      #242A3C;
  --border2:     #2E3550;
  --text:        #D8DEED;
  --text2:       #5A6285;
  --text3:       #343D5C;
  --accent:      #F0A94A;
  --good:        #4ADE80;
  --bad:         #E05A6E;

  --c-latin:     #E05A6E;
  --c-elec:      #3DD6F5;
  --c-rock:      #E8784A;
  --c-urban:     #9B7CF4;
  --c-folk:      #52B472;
  --c-classical: #E8C24A;
  --c-pop:       #5A9BE8;
  --c-soul:      #E87AC8;
  --c-world:     #4AB4A4;

  --r: 6px;
  --font: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono: "SF Mono","JetBrains Mono","Fira Mono",monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  background: var(--surface);
}
.logo { font-weight: 700; font-size: 13px; letter-spacing: .06em; color: var(--accent); text-transform: uppercase; }
.progress-wrap { flex: 1; display: flex; align-items: center; gap: 10px; }
.progress-bar { flex: 1; height: 3px; background: var(--border2); border-radius: 99px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 99px; transition: width .4s ease; }
.progress-label { font-variant-numeric: tabular-nums; color: var(--text2); font-size: 12px; white-space: nowrap; }
.btn-undo { background: none; border: 1px solid var(--border2); color: var(--text2); padding: 4px 10px; border-radius: var(--r); cursor: pointer; font-size: 12px; transition: all .15s; }
.btn-undo:hover { border-color: var(--text2); color: var(--text); }

main { display: flex; flex: 1; overflow: hidden; }

/* ── Player panel ── */
.player-panel {
  width: 272px; flex-shrink: 0;
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  padding: 20px; gap: 14px;
  background: var(--surface);
}
.now-playing-label { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--text3); }
.filename { font-family: var(--mono); font-size: 11px; color: var(--text); word-break: break-all; line-height: 1.5; min-height: 2.2em; }

.controls-row { display: flex; align-items: center; gap: 12px; }
.play-btn {
  width: 48px; height: 48px; border-radius: 50%;
  background: var(--accent); border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; transition: transform .12s;
}
.play-btn:hover { transform: scale(1.06); }
.play-btn svg { fill: #0F1117; }
.play-btn.playing .icon-play  { display: none; }
.play-btn:not(.playing) .icon-pause { display: none; }

.seek-wrap { flex: 1; height: 20px; display: flex; align-items: center; cursor: pointer; }
.seek-track { width: 100%; height: 3px; background: var(--border2); border-radius: 99px; overflow: hidden; }
.seek-fill { height: 100%; background: var(--accent); border-radius: 99px; width: 0%; pointer-events: none; }
.seek-wrap:hover .seek-track { height: 5px; }
.time-display { font-variant-numeric: tabular-nums; color: var(--text2); font-size: 12px; white-space: nowrap; }

/* ── Selected genres area ── */
.selected-area { display: flex; flex-direction: column; gap: 8px; flex: 1; }
.selected-label { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--text3); }
.selected-pills { display: flex; flex-wrap: wrap; gap: 5px; min-height: 28px; }
.pill {
  display: flex; align-items: center; gap: 5px;
  padding: 4px 8px; border-radius: 99px;
  background: var(--surface2); border: 1px solid var(--border2);
  font-size: 11px; color: var(--text);
}
.pill .pill-x { cursor: pointer; color: var(--text2); font-size: 13px; line-height: 1; }
.pill .pill-x:hover { color: var(--bad); }
.selected-hint { font-size: 11px; color: var(--text3); }

.confirm-btn {
  width: 100%; padding: 10px;
  background: var(--accent); border: none;
  color: #0F1117; border-radius: var(--r);
  cursor: pointer; font-size: 13px; font-weight: 600;
  transition: all .15s; opacity: .35; pointer-events: none;
}
.confirm-btn.ready { opacity: 1; pointer-events: auto; }
.confirm-btn.ready:hover { filter: brightness(1.08); }
.confirm-btn.ready:active { transform: scale(.98); }

.skip-btn {
  width: 100%; padding: 8px;
  background: none; border: 1px solid var(--border2);
  color: var(--text2); border-radius: var(--r);
  cursor: pointer; font-size: 12px; transition: all .15s;
}
.skip-btn:hover { border-color: var(--text); color: var(--text); }

.keyboard-hint { font-size: 10px; color: var(--text3); line-height: 1.9; }
.keyboard-hint kbd {
  display: inline-block; padding: 1px 5px;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: 3px; font-size: 10px; font-family: var(--mono); color: var(--text2);
}

.done-screen {
  display: none; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; text-align: center; flex: 1;
}
.done-screen code { font-family: var(--mono); font-size: 10px; color: var(--accent); }

/* ── Genre grid ── */
.genre-panel { flex: 1; overflow-y: auto; padding: 14px 18px; }
.genre-panel::-webkit-scrollbar { width: 4px; }
.genre-panel::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.genre-section { margin-bottom: 16px; }
.section-label {
  font-size: 9.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--text3); margin-bottom: 7px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.sdot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }

.genre-buttons { display: flex; flex-wrap: wrap; gap: 5px; }
.g {
  padding: 6px 12px; border-radius: var(--r);
  border: 1px solid var(--border2); background: var(--surface2);
  color: var(--text); font-size: 12px; cursor: pointer;
  transition: all .12s; white-space: nowrap; font-family: var(--font);
  user-select: none;
}
.g:hover { border-color: var(--c); background: color-mix(in srgb, var(--c) 12%, var(--surface2)); }
.g:active { transform: scale(.94); }
.g.selected {
  border-color: var(--c);
  background: color-mix(in srgb, var(--c) 22%, var(--surface2));
  color: #fff;
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--c) 40%, transparent);
}
.g.suggested {
  border-color: var(--c);
  border-style: dashed;
  background: color-mix(in srgb, var(--c) 22%, var(--surface2));
  color: #fff;
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--c) 40%, transparent);
}
.g.flash { animation: gbflash .35s ease; }

.ai-hint {
  display: none;
  font-size: 11px; color: var(--text2);
  background: color-mix(in srgb, var(--accent) 8%, var(--surface));
  border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
  border-radius: var(--r); padding: 5px 10px; margin-top: 4px;
}
.ai-hint.visible { display: block; }
@keyframes gbflash {
  0%   { background: color-mix(in srgb, var(--c) 55%, var(--surface2)); color: #fff; }
  100% { background: color-mix(in srgb, var(--c) 22%, var(--surface2)); }
}

.toast {
  position: fixed; bottom: 20px; right: 20px;
  background: var(--surface); border: 1px solid var(--border2);
  color: var(--text); padding: 8px 14px; border-radius: var(--r);
  font-size: 12px; opacity: 0; transform: translateY(8px);
  transition: all .2s; pointer-events: none; z-index: 99;
}
.toast.show { opacity: 1; transform: none; }
.toast.ok  { border-color: var(--good); }
.toast.err { border-color: var(--bad); }
</style>
</head>
<body>

<header>
  <span class="logo">Label Tool</span>
  <div class="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <span class="progress-label" id="progressLabel">— / —</span>
  </div>
  <button class="btn-undo" onclick="undo()">↩ Undo</button>
</header>

<main>
  <!-- Player -->
  <div class="player-panel" id="playerPanel">
    <div>
      <div class="now-playing-label">Now playing</div>
      <div class="filename" id="filename">Loading…</div>
      <div class="ai-hint" id="aiHint"></div>
    </div>

    <div class="controls-row">
      <button class="play-btn" id="playBtn" onclick="togglePlay()">
        <svg class="icon-play" width="18" height="18" viewBox="0 0 20 20"><polygon points="5,3 17,10 5,17"/></svg>
        <svg class="icon-pause" width="18" height="18" viewBox="0 0 20 20"><rect x="4" y="3" width="4" height="14" rx="1"/><rect x="12" y="3" width="4" height="14" rx="1"/></svg>
      </button>
      <div class="seek-wrap" onclick="seek(event)">
        <div class="seek-track"><div class="seek-fill" id="seekFill"></div></div>
      </div>
      <span class="time-display" id="timeDisplay">0:00</span>
    </div>

    <div class="selected-area">
      <div class="selected-label">Selected genres <span id="selCount" style="color:var(--text2)">(0 / 3)</span></div>
      <div class="selected-pills" id="selectedPills">
        <span class="selected-hint" id="selHint">Click genres on the right →</span>
      </div>
    </div>

    <button class="confirm-btn" id="confirmBtn" onclick="confirmLabel()">Label &amp; Next →</button>
    <button class="skip-btn" onclick="skip()">Skip — not sure</button>

    <div class="keyboard-hint">
      <kbd>Space</kbd> play/pause &nbsp; <kbd>→</kbd> skip<br>
      <kbd>Enter</kbd> confirm &nbsp; <kbd>⌘Z</kbd> undo
    </div>

    <div class="done-screen" id="doneScreen">
      <div style="font-size:32px">✓</div>
      <strong>All done!</strong>
      <p style="color:var(--text2);font-size:11px;line-height:1.6">Run the ingestion script:</p>
      <code>python3 ingest_labeled_audio.py labeled_audio</code>
    </div>
  </div>

  <!-- Genre grid -->
  <div class="genre-panel">

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-latin)"></span>Latin</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Latin')">Latin</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Salsa')">Salsa</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Bachata')">Bachata</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Merengue')">Merengue</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Cumbia')">Cumbia</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Tango')">Tango</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Reggaetón')">Reggaetón</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Samba')">Samba</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Bossa Nova')">Bossa Nova</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Mambo')">Mambo</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Cha-Cha')">Cha-Cha</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Flamenco')">Flamenco</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Bolero')">Bolero</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Latin Folk')">Latin Folk</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Trap Latino')">Trap Latino</button>
        <button class="g" style="--c:var(--c-latin)" onclick="toggle(this,'Urbano')">Urbano</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-elec)"></span>Electronic</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Electronic')">Electronic</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'EDM')">EDM</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'House')">House</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Techno')">Techno</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Trance')">Trance</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Dubstep')">Dubstep</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Drum and Bass')">Drum &amp; Bass</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Synthwave')">Synthwave</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'Synth-Pop')">Synth-Pop</button>
        <button class="g" style="--c:var(--c-elec)" onclick="toggle(this,'New Wave')">New Wave</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-rock)"></span>Rock / Metal</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Rock')">Rock</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Alternative Rock')">Alternative Rock</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Hard Rock')">Hard Rock</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Progressive Rock')">Progressive Rock</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Punk')">Punk</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Grunge')">Grunge</button>
        <button class="g" style="--c:var(--c-rock)" onclick="toggle(this,'Metal')">Metal</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-urban)"></span>Hip-Hop / Urban</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-urban)" onclick="toggle(this,'Hip-Hop')">Hip-Hop</button>
        <button class="g" style="--c:var(--c-urban)" onclick="toggle(this,'Trap')">Trap</button>
        <button class="g" style="--c:var(--c-urban)" onclick="toggle(this,'Reggae')">Reggae</button>
        <button class="g" style="--c:var(--c-urban)" onclick="toggle(this,'Dancehall')">Dancehall</button>
        <button class="g" style="--c:var(--c-urban)" onclick="toggle(this,'Ska')">Ska</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-soul)"></span>Soul / Jazz / R&amp;B</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Jazz')">Jazz</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Swing')">Swing</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Big Band')">Big Band</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'R&B')">R&amp;B</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Soul')">Soul</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Funk/Soul')">Funk / Soul</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Disco')">Disco</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Gospel')">Gospel</button>
        <button class="g" style="--c:var(--c-soul)" onclick="toggle(this,'Blues')">Blues</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-folk)"></span>Folk / Acoustic / Country</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-folk)" onclick="toggle(this,'Folk')">Folk</button>
        <button class="g" style="--c:var(--c-folk)" onclick="toggle(this,'Acoustic')">Acoustic</button>
        <button class="g" style="--c:var(--c-folk)" onclick="toggle(this,'Country')">Country</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-pop)"></span>Pop / Indie</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'Pop')">Pop</button>
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'Indie')">Indie</button>
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'K-Pop')">K-Pop</button>
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'Lo-Fi')">Lo-Fi</button>
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'HyperPop')">HyperPop</button>
        <button class="g" style="--c:var(--c-pop)" onclick="toggle(this,'Ballad')">Ballad</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-classical)"></span>Classical / Film</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Classical')">Classical</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Baroque')">Baroque</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Opera')">Opera</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Waltz')">Waltz</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Film Score')">Film Score</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Cinematic')">Cinematic</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Musical Theatre')">Musical Theatre</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'Ambient')">Ambient</button>
        <button class="g" style="--c:var(--c-classical)" onclick="toggle(this,'New Age')">New Age</button>
      </div>
    </div>

    <div class="genre-section">
      <div class="section-label"><span class="sdot" style="background:var(--c-world)"></span>World / Other</div>
      <div class="genre-buttons">
        <button class="g" style="--c:var(--c-world)" onclick="toggle(this,'Afrobeats')">Afrobeats</button>
        <button class="g" style="--c:var(--c-world)" onclick="toggle(this,'World Music')">World Music</button>
        <button class="g" style="--c:var(--c-world)" onclick="toggle(this,'Corporate')">Corporate</button>
        <button class="g" style="--c:var(--c-world)" onclick="toggle(this,'Childrens')">Children's</button>
      </div>
    </div>

  </div><!-- end genre-panel -->
</main>

<div class="toast" id="toast"></div>
<audio id="audio" preload="auto"></audio>

<script>
const audio    = document.getElementById('audio');
const playBtn  = document.getElementById('playBtn');
const seekFill = document.getElementById('seekFill');
const timeDis  = document.getElementById('timeDisplay');
const fileEl   = document.getElementById('filename');
const progFill = document.getElementById('progressFill');
const progLab  = document.getElementById('progressLabel');
const doneScr  = document.getElementById('doneScreen');
const confirmBtn = document.getElementById('confirmBtn');
const pillsEl  = document.getElementById('selectedPills');
const selHint  = document.getElementById('selHint');
const selCount = document.getElementById('selCount');
const aiHint   = document.getElementById('aiHint');

let files   = [];
let total   = 0;
let current = null;
let busy    = false;
let hints   = {};   // filename → {prediction, confidence, top3}

// Multi-select state: genre string → button element
const selected = new Map();

function fmt(s) {
  if (isNaN(s)) return '0:00';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}

audio.addEventListener('timeupdate', () => {
  if (!audio.duration) return;
  seekFill.style.width = (audio.currentTime / audio.duration * 100) + '%';
  timeDis.textContent = fmt(audio.currentTime) + ' / ' + fmt(audio.duration);
});
audio.addEventListener('play',  () => playBtn.classList.add('playing'));
audio.addEventListener('pause', () => playBtn.classList.remove('playing'));
audio.addEventListener('ended', () => playBtn.classList.remove('playing'));

function togglePlay() {
  if (audio.paused) audio.play(); else audio.pause();
}

function seek(e) {
  if (!audio.duration) return;
  const r = e.currentTarget.getBoundingClientRect();
  audio.currentTime = (e.clientX - r.left) / r.width * audio.duration;
}

// ── Multi-select ─────────────────────────────────────────────────────────────
let aiSuggestion = null;   // currently suggested genre name

function toggle(btn, genre) {
  if (selected.has(genre)) {
    selected.delete(genre);
    btn.classList.remove('selected');
    btn.classList.remove('suggested');
    if (genre === aiSuggestion) aiSuggestion = null;
  } else {
    if (selected.size >= 3) {
      showToast('Max 3 genres per track', 'err'); return;
    }
    // If clicking a different genre, auto-clear the AI suggestion
    if (aiSuggestion && genre !== aiSuggestion) {
      const sugBtn = selected.get(aiSuggestion);
      if (sugBtn) {
        sugBtn.classList.remove('selected');
        sugBtn.classList.remove('suggested');
        selected.delete(aiSuggestion);
      }
      aiSuggestion = null;
      aiHint.className = 'ai-hint';
    }
    selected.set(genre, btn);
    btn.classList.add('selected');
  }
  renderPills();
}

function renderPills() {
  pillsEl.innerHTML = '';
  if (selected.size === 0) {
    pillsEl.appendChild(selHint);
    selHint.style.display = '';
  } else {
    selHint.style.display = 'none';
    selected.forEach((btn, genre) => {
      const pill = document.createElement('div');
      pill.className = 'pill';
      pill.innerHTML = `<span>${genre}</span><span class="pill-x" onclick="toggle(document.querySelector('[data-g=\\'${genre}\\']'),'${genre}')">×</span>`;
      pillsEl.appendChild(pill);
    });
  }
  const n = selected.size;
  selCount.textContent = `(${n} / 3)`;
  confirmBtn.classList.toggle('ready', n > 0);
}

// attach data-g to all genre buttons for pill removal lookup
document.querySelectorAll('.g').forEach(b => {
  const m = b.getAttribute('onclick').match(/'([^']+)'\)$/);
  if (m) b.dataset.g = m[1];
});

function clearSelection() {
  selected.forEach((btn) => btn.classList.remove('selected'));
  selected.clear();
  renderPills();
}

// ── Labeling ──────────────────────────────────────────────────────────────────
function confirmLabel() {
  if (selected.size === 0 || busy) return;
  busy = true;

  const genres = [...selected.keys()];
  // flash selected buttons
  selected.forEach(btn => {
    btn.classList.add('flash');
    setTimeout(() => btn.classList.remove('flash'), 380);
  });

  fetch('/api/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: current, genres })
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      showToast('→ ' + genres.join(' + '), 'ok');
      clearSelection();
      setTimeout(() => { busy = false; advance(); }, 300);
    } else {
      showToast('Error: ' + (d.error || 'unknown'), 'err');
      busy = false;
    }
  })
  .catch(() => { showToast('Network error', 'err'); busy = false; });
}

function skip() {
  if (busy) return;
  busy = true;
  clearSelection();
  fetch('/api/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: current, genres: ['_unsure'] })
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) { showToast('Skipped', ''); setTimeout(() => { busy = false; advance(); }, 200); }
    else { showToast(d.error || 'Error', 'err'); busy = false; }
  })
  .catch(() => { busy = false; });
}

function advance() {
  files.shift();
  const done = total - files.length;
  progFill.style.width = (total > 0 ? done / total * 100 : 0) + '%';
  progLab.textContent = files.length + ' remaining';
  if (files.length === 0) showDone();
  else loadFile(files[0]);
}

function loadFile(name) {
  current = name;
  fileEl.textContent = name;
  audio.src = '/audio/' + encodeURIComponent(name);
  audio.load();
  audio.play().catch(() => {});
  applySuggestion(name);
}

function applySuggestion(name) {
  document.querySelectorAll('.g.suggested').forEach(b => b.classList.remove('suggested'));
  aiHint.className = 'ai-hint';
  aiSuggestion = null;

  const h = hints[name];
  if (!h || !h.prediction) return;

  const btn = document.querySelector(`.g[data-g="${CSS.escape(h.prediction)}"]`);
  if (btn) {
    btn.classList.add('suggested');
    aiSuggestion = h.prediction;
    if (!selected.has(h.prediction)) {
      selected.set(h.prediction, btn);
      renderPills();
    }
  }

  const pct = Math.round((h.confidence || 0) * 100);
  const top3 = (h.top3 || []).map(([g, c]) => `${g} ${Math.round(c*100)}%`).join(' · ');
  aiHint.textContent = `AI: ${h.prediction} (${pct}%)  —  ${top3}`;
  aiHint.className = 'ai-hint visible';
}

function showDone() {
  audio.pause();
  fileEl.textContent = '';
  doneScr.style.display = 'flex';
  document.querySelector('.controls-row').style.visibility = 'hidden';
  document.querySelector('.selected-area').style.display = 'none';
  confirmBtn.style.display = 'none';
  document.querySelector('.skip-btn').style.display = 'none';
}

function undo() {
  fetch('/api/undo', { method: 'POST' })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      fetch('/api/files').then(r => r.json()).then(list => {
        files = list;
        progFill.style.width = (total > 0 ? (total - list.length) / total * 100 : 0) + '%';
        progLab.textContent = list.length + ' remaining';
        if (list.length > 0) {
          doneScr.style.display = 'none';
          document.querySelector('.controls-row').style.visibility = '';
          document.querySelector('.selected-area').style.display = '';
          confirmBtn.style.display = '';
          document.querySelector('.skip-btn').style.display = '';
          loadFile(list[0]);
        }
        showToast('↩ Restored ' + d.restored, 'ok');
      });
    } else { showToast(d.error || 'Nothing to undo', 'err'); }
  });
}

let toastTimer;
function showToast(msg, cls) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'toast show ' + (cls || '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = 'toast', 1800);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'BUTTON') return;
  if (e.code === 'Space')  { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowRight') skip();
  if (e.code === 'Enter')  { e.preventDefault(); confirmLabel(); }
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') { e.preventDefault(); undo(); }
});

// Boot — load file list and AI hints in parallel
Promise.all([
  fetch('/api/files').then(r => r.json()),
  fetch('/api/hint').then(r => r.json()).catch(() => ({}))
]).then(([list, hintData]) => {
  files = list; total = list.length; hints = hintData;
  progFill.style.width = '0%';
  progLab.textContent = list.length + ' remaining';
  if (list.length === 0) fileEl.textContent = 'No audio files found in source folder.';
  else loadFile(list[0]);
}).catch(() => { fileEl.textContent = 'Could not connect to server.'; });

renderPills();
</script>
</body>
</html>"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ('/', '/index.html'):  self._send(200, 'text/html; charset=utf-8', HTML.encode())
        elif p == '/api/files':        self._list_files()
        elif p == '/api/hint':         self._list_hints()
        elif p.startswith('/audio/'): self._serve_audio(urllib.parse.unquote(p[7:]))
        else:                          self.send_error(404)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        if   p == '/api/label': self._label(body)
        elif p == '/api/undo':  self._undo()
        else:                   self.send_error(404)

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', len(data))
        self.end_headers()
        self.wfile.write(data)

    def _list_files(self):
        try:
            existing = {f for f in os.listdir(SOURCE_DIR)
                        if Path(f).suffix.lower() in AUDIO_EXTS}
        except Exception:
            existing = set()
        if _presort:
            # Return presort order (filtered to files still in source dir)
            names = [r['filename'] for r in _presort.values() if r['filename'] in existing]
            # Append any new files not in presort (alphabetical at the end)
            names += sorted(f for f in existing if f not in {r['filename'] for r in _presort.values()})
        else:
            names = sorted(existing)
        self._send(200, 'application/json', json.dumps(names).encode())

    def _list_hints(self):
        self._send(200, 'application/json', json.dumps(_presort).encode())

    def _serve_audio(self, filename):
        fp = os.path.normpath(os.path.join(SOURCE_DIR, filename))
        if not fp.startswith(os.path.normpath(SOURCE_DIR) + os.sep):
            self.send_error(403); return
        if not os.path.isfile(fp):
            self.send_error(404); return
        size = os.path.getsize(fp)
        mime = mimetypes.guess_type(fp)[0] or 'audio/mpeg'
        rng  = self.headers.get('Range', '')
        if rng:
            try:
                r = rng.replace('bytes=', '').split('-')
                s, e = int(r[0]), int(r[1]) if r[1] else size - 1
                ln   = e - s + 1
                self.send_response(206)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Range', f'bytes {s}-{e}/{size}')
                self.send_header('Content-Length', ln)
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()
                with open(fp, 'rb') as f: f.seek(s); self.wfile.write(f.read(ln))
            except Exception as ex: self.send_error(500, str(ex))
        else:
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', size)
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            with open(fp, 'rb') as f: self.wfile.write(f.read())

    def _label(self, body):
        filename = body.get('filename', '')
        genres   = body.get('genres', [])
        if not filename or not genres:
            self.send_error(400); return

        src = os.path.normpath(os.path.join(SOURCE_DIR, filename))
        if not src.startswith(os.path.normpath(SOURCE_DIR) + os.sep):
            self.send_error(403); return
        if not os.path.isfile(src):
            self._send(404, 'application/json',
                       json.dumps({'ok': False, 'error': 'File not found'}).encode()); return

        def unique_dst(genre):
            d = os.path.join(OUTPUT_DIR, genre)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, filename)
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                i = 1
                while os.path.exists(dst):
                    dst = os.path.join(d, f'{base}_{i}{ext}'); i += 1
            return dst

        try:
            primary_dst    = unique_dst(genres[0])
            secondary_dsts = [unique_dst(g) for g in genres[1:]]

            # Move source to primary genre folder
            shutil.move(src, primary_dst)
            # Copy to any additional genre folders
            for dst in secondary_dsts:
                shutil.copy2(primary_dst, dst)

            _history.append((primary_dst, src, secondary_dsts))
            self._send(200, 'application/json', json.dumps({'ok': True}).encode())
        except Exception as ex:
            self._send(500, 'application/json',
                       json.dumps({'ok': False, 'error': str(ex)}).encode())

    def _undo(self):
        if not _history:
            self._send(200, 'application/json',
                       json.dumps({'ok': False, 'error': 'Nothing to undo'}).encode()); return
        primary_dst, src, secondary_dsts = _history.pop()
        # Remove secondary copies
        for dst in secondary_dsts:
            try: os.remove(dst)
            except FileNotFoundError: pass
        # Move primary back to source
        if os.path.isfile(primary_dst):
            shutil.move(primary_dst, src)
            self._send(200, 'application/json',
                       json.dumps({'ok': True, 'restored': os.path.basename(src)}).encode())
        else:
            self._send(200, 'application/json',
                       json.dumps({'ok': False, 'error': 'File no longer exists'}).encode())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    SOURCE_DIR = os.path.abspath(sys.argv[1])
    OUTPUT_DIR = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 \
                 else os.path.join(os.getcwd(), 'labeled_audio')

    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: '{SOURCE_DIR}' is not a directory."); sys.exit(1)

    # Load presort data if available
    presort_path = os.path.join(SOURCE_DIR, '.presort.json')
    if os.path.exists(presort_path):
        with open(presort_path) as f:
            presort_list = json.load(f)
        _presort.update({r['filename']: r for r in presort_list})
        print(f"  Loaded presort: {len(_presort)} predictions")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = sum(1 for f in os.listdir(SOURCE_DIR) if Path(f).suffix.lower() in AUDIO_EXTS)

    print(f"""
  Genre Label Tool
  ─────────────────────────────────────────
  Source:  {SOURCE_DIR}
           {n} audio files found
  Output:  {OUTPUT_DIR}
  ─────────────────────────────────────────
  Open → http://localhost:{PORT}
  Ctrl+C to stop.
""")
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    server = HTTPServer(('127.0.0.1', PORT), Handler)
    try:    server.serve_forever()
    except KeyboardInterrupt: print('\nStopped.')
