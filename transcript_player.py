"""
transcript_player.py
────────────────────────────────────────────────────────────
Composant Streamlit : player audio interactif + transcript
avec timestamps, couleurs Agent/Customer, et seek au clic.

Usage dans dashboard_v2.py :
    from transcript_player import show_transcript_player
    show_transcript_player(audio_bytes, transcript, diarization)
"""

import base64
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

def show_transcript_player(
    audio_bytes:   bytes,
    transcript_text: str,
    segments:      list,          # list of Segment objects or dicts
    turns:         list = None,   # list of DiarizedTurn objects or dicts
    audio_format:  str = "audio/mp3",
):
    """
    Affiche un player audio interactif + transcript visuel.

    Parameters:
        audio_bytes    : contenu binaire du fichier audio
        transcript_text: texte complet de la transcription
        segments       : liste de segments Whisper avec timestamps
        turns          : liste de tours diarisés (agent/customer)
        audio_format   : format MIME de l'audio
    """

    st.markdown("### Transcription de l'appel")

    # ── Encode audio en base64 ────────────────────────────────────────────────
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # ── Prépare les données de segments ──────────────────────────────────────
    seg_data = _build_segments_data(segments, turns)

    # ── Injecte dans le composant HTML ────────────────────────────────────────
    html = _build_player_html(audio_b64, audio_format, seg_data, transcript_text)
    components.html(html, height=520, scrolling=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_segments_data(segments: list, turns: list = None) -> list:
    """
    Convertit les segments Whisper + turns pyannote en liste JSON
    compatible avec le player HTML.
    """
    # Build speaker lookup: start_sec -> role
    speaker_lookup = {}
    if turns:
        for turn in turns:
            if hasattr(turn, "start_sec"):
                speaker_lookup[(turn.start_sec, turn.end_sec)] = {
                    "speaker": getattr(turn, "speaker", "SPEAKER_00"),
                    "role":    getattr(turn, "role",    "unknown"),
                }
            elif isinstance(turn, dict):
                speaker_lookup[(turn.get("start_sec", 0),
                                turn.get("end_sec", 0))] = {
                    "speaker": turn.get("speaker", "SPEAKER_00"),
                    "role":    turn.get("role",    "unknown"),
                }

    def get_role(seg_start: float) -> tuple:
        """Find the role for a segment by timestamp overlap."""
        for (start, end), info in speaker_lookup.items():
            if start <= seg_start <= end:
                return info["speaker"], info["role"]
        return "SPEAKER_00", "unknown"

    result = []
    for seg in segments:
        # Support both objects and dicts
        if hasattr(seg, "start_sec"):
            start = seg.start_sec
            end   = seg.end_sec
            text  = seg.text
            words = getattr(seg, "words", [])
        elif isinstance(seg, dict):
            start = seg.get("start_sec", seg.get("start", 0))
            end   = seg.get("end_sec",   seg.get("end",   0))
            text  = seg.get("text", "")
            words = seg.get("words", [])
        else:
            continue

        speaker, role = get_role(start)

        # Build word-level data
        word_data = []
        for w in words:
            if hasattr(w, "word"):
                word_data.append({
                    "word":  w.word,
                    "start": w.start_sec,
                    "end":   w.end_sec,
                })
            elif isinstance(w, dict):
                word_data.append({
                    "word":  w.get("word", ""),
                    "start": w.get("start_sec", start),
                    "end":   w.get("end_sec",   end),
                })

        result.append({
            "start":   round(start, 2),
            "end":     round(end, 2),
            "text":    text.strip(),
            "speaker": speaker,
            "role":    role,
            "words":   word_data,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

def _build_player_html(
    audio_b64:    str,
    audio_format: str,
    segments:     list,
    full_text:    str,
) -> str:

    segments_json = json.dumps(segments, ensure_ascii=False)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0e1117;
    color: #e6edf3;
    padding: 12px;
  }}

  /* ── PLAYER ─────────────────────────────────────────────── */
  .player-wrap {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }}

  audio {{
    width: 100%;
    height: 36px;
    outline: none;
    border-radius: 6px;
    margin-bottom: 8px;
  }}

  audio::-webkit-media-controls-panel {{
    background: #21262d;
  }}

  .time-display {{
    font-family: monospace;
    font-size: 12px;
    color: #8b949e;
    text-align: right;
  }}

  /* ── CONTROLS ───────────────────────────────────────────── */
  .controls {{
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
    align-items: center;
  }}

  .btn {{
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 4px 12px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
  }}
  .btn:hover {{ background: #30363d; }}
  .btn.active {{ background: #1f6feb; border-color: #1f6feb; }}

  .legend {{
    display: flex;
    gap: 12px;
    margin-left: auto;
    align-items: center;
  }}

  .legend-dot {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #8b949e;
  }}

  .dot {{
    width: 10px; height: 10px; border-radius: 50%;
  }}

  /* ── TRANSCRIPT ─────────────────────────────────────────── */
  .transcript-wrap {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
    max-height: 320px;
    overflow-y: auto;
    scroll-behavior: smooth;
  }}

  .transcript-wrap::-webkit-scrollbar {{ width: 6px; }}
  .transcript-wrap::-webkit-scrollbar-track {{ background: #0e1117; }}
  .transcript-wrap::-webkit-scrollbar-thumb {{
    background: #30363d; border-radius: 3px;
  }}

  /* ── SEGMENT ────────────────────────────────────────────── */
  .segment {{
    display: grid;
    grid-template-columns: 90px 1fr;
    gap: 10px;
    padding: 8px 6px;
    border-radius: 7px;
    margin-bottom: 4px;
    transition: background 0.2s;
    cursor: pointer;
  }}

  .segment:hover {{ background: #21262d; }}

  .segment.active {{
    background: rgba(31, 111, 235, 0.12);
    border-left: 3px solid #1f6feb;
  }}

  .seg-meta {{
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    padding-top: 2px;
    gap: 3px;
  }}

  .seg-time {{
    font-family: monospace;
    font-size: 10px;
    color: #484f58;
    background: #21262d;
    padding: 1px 5px;
    border-radius: 3px;
    white-space: nowrap;
  }}

  .role-badge {{
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 1px 6px;
    border-radius: 3px;
    white-space: nowrap;
  }}

  .role-agent    {{ background: rgba(31,111,235,0.2); color: #58a6ff; }}
  .role-customer {{ background: rgba(63,185,80,0.2);  color: #3fb950; }}
  .role-unknown  {{ background: rgba(139,148,158,0.2);color: #8b949e; }}

  .seg-text {{
    font-size: 13.5px;
    line-height: 1.6;
    color: #c9d1d9;
    padding-top: 1px;
  }}

  /* ── WORD HIGHLIGHTING ──────────────────────────────────── */
  .word {{
    display: inline;
    padding: 1px 1px;
    border-radius: 2px;
    transition: background 0.1s, color 0.1s;
    cursor: pointer;
  }}

  .word:hover {{
    background: rgba(31,111,235,0.25);
    color: #79c0ff;
  }}

  .word.highlight {{
    background: rgba(31,111,235,0.4);
    color: #ffffff;
    border-radius: 2px;
  }}

  /* ── NO DATA ────────────────────────────────────────────── */
  .no-data {{
    padding: 20px;
    text-align: center;
    color: #484f58;
    font-size: 13px;
  }}
</style>
</head>
<body>

<!-- PLAYER -->
<div class="player-wrap">
  <audio id="audioPlayer" controls preload="auto">
    <source src="data:{audio_format};base64,{audio_b64}" type="{audio_format}">
  </audio>
  <div class="time-display" id="timeDisplay">0:00 / 0:00</div>
</div>

<!-- CONTROLS -->
<div class="controls">
  <button class="btn" onclick="setSpeed(0.75)">0.75x</button>
  <button class="btn active" id="btn1x" onclick="setSpeed(1.0)">1x</button>
  <button class="btn" onclick="setSpeed(1.25)">1.25x</button>
  <button class="btn" onclick="setSpeed(1.5)">1.5x</button>
  <button class="btn" onclick="toggleAutoScroll()" id="btnScroll">
    Auto-scroll ON
  </button>
  <div class="legend">
    <span class="legend-dot"><span class="dot" style="background:#58a6ff"></span>Agent</span>
    <span class="legend-dot"><span class="dot" style="background:#3fb950"></span>Customer</span>
    <span class="legend-dot"><span class="dot" style="background:#8b949e"></span>Unknown</span>
  </div>
</div>

<!-- TRANSCRIPT -->
<div class="transcript-wrap" id="transcriptWrap">
</div>

<script>
const audio      = document.getElementById('audioPlayer');
const timeDisp   = document.getElementById('timeDisplay');
const wrap       = document.getElementById('transcriptWrap');
const SEGMENTS   = {segments_json};
let autoScroll   = true;
let activeSegIdx = -1;
let activeWordEl = null;

// ── Build transcript DOM ───────────────────────────────────────────────────
function buildTranscript() {{
  if (!SEGMENTS || SEGMENTS.length === 0) {{
    wrap.innerHTML = '<div class="no-data">No segment data available.<br>Enable word_timestamps in config.yaml for word-level highlighting.</div>';
    return;
  }}

  SEGMENTS.forEach((seg, si) => {{
    const div = document.createElement('div');
    div.className = 'segment';
    div.id = 'seg-' + si;
    div.onclick = () => seekTo(seg.start);

    // Meta column
    const meta = document.createElement('div');
    meta.className = 'seg-meta';

    const timeSpan = document.createElement('span');
    timeSpan.className = 'seg-time';
    timeSpan.textContent = formatTime(seg.start) + ' - ' + formatTime(seg.end);

    const roleSpan = document.createElement('span');
    roleSpan.className = 'role-badge role-' + (seg.role || 'unknown');
    roleSpan.textContent = seg.role === 'agent'    ? 'Agent' :
                           seg.role === 'customer' ? 'Customer' : 'Unknown';

    meta.appendChild(roleSpan);
    meta.appendChild(timeSpan);

    // Text column
    const textDiv = document.createElement('div');
    textDiv.className = 'seg-text';
    textDiv.id = 'segtext-' + si;

    if (seg.words && seg.words.length > 0) {{
      // Word-level spans for highlighting
      seg.words.forEach((w, wi) => {{
        const span = document.createElement('span');
        span.className = 'word';
        span.id = 'w-' + si + '-' + wi;
        span.textContent = w.word;
        span.onclick = (e) => {{ e.stopPropagation(); seekTo(w.start); }};
        span.title = formatTime(w.start);
        textDiv.appendChild(span);
      }});
    }} else {{
      // Segment-level only
      textDiv.textContent = seg.text;
    }}

    div.appendChild(meta);
    div.appendChild(textDiv);
    wrap.appendChild(div);
  }});
}}

// ── Time update handler ────────────────────────────────────────────────────
audio.addEventListener('timeupdate', () => {{
  const t = audio.currentTime;

  // Update time display
  timeDisp.textContent = formatTime(t) + ' / ' + formatTime(audio.duration || 0);

  // Find active segment
  let found = -1;
  for (let i = 0; i < SEGMENTS.length; i++) {{
    if (t >= SEGMENTS[i].start && t <= SEGMENTS[i].end) {{
      found = i; break;
    }}
  }}

  if (found !== activeSegIdx) {{
    // Deactivate old
    if (activeSegIdx >= 0) {{
      const old = document.getElementById('seg-' + activeSegIdx);
      if (old) old.classList.remove('active');
    }}
    // Activate new
    if (found >= 0) {{
      const el = document.getElementById('seg-' + found);
      if (el) {{
        el.classList.add('active');
        if (autoScroll) {{
          el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }}
      }}
    }}
    activeSegIdx = found;
  }}

  // Word-level highlight
  if (found >= 0 && SEGMENTS[found].words && SEGMENTS[found].words.length > 0) {{
    const words = SEGMENTS[found].words;
    for (let wi = 0; wi < words.length; wi++) {{
      if (t >= words[wi].start && t <= words[wi].end) {{
        const el = document.getElementById('w-' + found + '-' + wi);
        if (el && el !== activeWordEl) {{
          if (activeWordEl) activeWordEl.classList.remove('highlight');
          el.classList.add('highlight');
          activeWordEl = el;
        }}
        break;
      }}
    }}
  }}
}});

// ── Helpers ────────────────────────────────────────────────────────────────
function seekTo(t) {{
  audio.currentTime = t;
  if (audio.paused) audio.play();
}}

function formatTime(s) {{
  if (!s || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + String(sec).padStart(2, '0');
}}

function setSpeed(s) {{
  audio.playbackRate = s;
  document.querySelectorAll('.btn').forEach(b => {{
    if (b.textContent === s + 'x' || b.textContent === '1x' && s === 1.0) {{
      b.classList.toggle('active', parseFloat(b.textContent) === s ||
                                   (b.id === 'btn1x' && s === 1.0));
    }}
  }});
  document.querySelectorAll('.controls .btn').forEach(b => {{
    const spd = parseFloat(b.textContent);
    if (!isNaN(spd)) b.classList.toggle('active', spd === s);
  }});
}}

function toggleAutoScroll() {{
  autoScroll = !autoScroll;
  document.getElementById('btnScroll').textContent =
    'Auto-scroll ' + (autoScroll ? 'ON' : 'OFF');
  document.getElementById('btnScroll').classList.toggle('active', autoScroll);
}}

// ── Init ───────────────────────────────────────────────────────────────────
buildTranscript();
</script>
</body>
</html>
"""
