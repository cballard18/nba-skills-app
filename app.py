from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

app = Flask(__name__)

# Load once at startup
print("Loading data...")
df = pd.read_parquet("datasets/skills.parquet")
df["game_date"] = pd.to_datetime(df["game_date"])

ff = pd.read_parquet("datasets/four_factor_skills.parquet")
ff["game_date"] = pd.to_datetime(ff["game_date"])
df = df.merge(ff, on=["player", "game_date"], how="left")

# Scale four-factor APM coefficients to per-100-possession units.
# Negate otov/dtov so positive = good (fewer own TOs / more forced TOs).
FF_COLS_POS = ["oefg", "oorb", "oftr", "defg", "dorb", "dftr"]
FF_COLS_NEG = ["otov", "dtov"]
for col in FF_COLS_POS:
    df[col] = df[col] * 100
for col in FF_COLS_NEG:
    df[col] = -df[col] * 100

df = df.sort_values(["player", "game_date"])
df = df[df.groupby("player")["player"].transform("count") >= 10]
PLAYERS = sorted(df["player"].unique().tolist())
print(f"Loaded {len(df)} rows, {len(PLAYERS)} players")

spm_ts = pd.read_parquet("datasets/spm_ts.parquet")
spm_ts["game_date"] = pd.to_datetime(spm_ts["game_date"])
spm_ts = spm_ts.sort_values(["player", "game_date"])

spm_career = pd.read_parquet("datasets/spm_career.parquet")
spm_career = spm_career.sort_values("spm", ascending=False).reset_index(drop=True)
SPM_PLAYERS = sorted(spm_ts["player"].unique().tolist())
spm_n_games = spm_ts.groupby("player").size().to_dict()
SPM_SKILLS  = {"ospm": "O-SPM", "dspm": "D-SPM", "spm": "SPM"}
print(f"SPM: {len(spm_ts)} ts rows, {len(spm_career)} career rows")

# Precompute league averages (latest estimate per player → mean)
_latest_skills = df.groupby("player").last().reset_index()
LEAGUE_AVG = {col: round(float(_latest_skills[col].mean()), 4)
              for col in _latest_skills.columns if col not in ("player", "game_date")}
for skill in SPM_SKILLS:
    if skill in spm_career.columns:
        LEAGUE_AVG[skill] = round(float(spm_career[skill].mean()), 4)
del _latest_skills

SKILLS = {
    "pts_per100":        "Points per 100",
    "ast_per100":        "Assists per 100",
    "dreb_per100":       "Def Rebounds per 100",
    "oreb_per100":       "Off Rebounds per 100",
    "tov_per100":        "Turnovers per 100",
    "blk_per100":        "Blocks per 100",
    "stl_per100":        "Steals per 100",
    "pf_per100":         "Fouls per 100",
    "fg2_pct":           "2PT FG%",
    "fg3_pct":           "3PT FG%",
    "ft_pct":            "FT%",
    "fg2a_rate_per100":  "2PT Attempts per 100",
    "fg3a_rate_per100":  "3PT Attempts per 100",
    "fta_rate_per100":   "FT Attempts per 100",
    "usage_per100":      "Usage Rate",
    "oefg":              "Off eFG% Impact",
    "otov":              "Off Ball Security",
    "oorb":              "Off Reb Impact",
    "oftr":              "Off FT Rate Impact",
    "defg":              "Def eFG% Impact",
    "dtov":              "Def Forced TO Impact",
    "dorb":              "Def Reb Impact",
    "dftr":              "Def FT Rate Impact",
}

SKILL_ABBR = {
    "pts_per100":        "PTS/100",
    "ast_per100":        "AST/100",
    "dreb_per100":       "DREB/100",
    "oreb_per100":       "OREB/100",
    "tov_per100":        "TOV/100",
    "blk_per100":        "BLK/100",
    "stl_per100":        "STL/100",
    "pf_per100":         "PF/100",
    "fg2_pct":           "FG2%",
    "fg3_pct":           "FG3%",
    "ft_pct":            "FT%",
    "fg2a_rate_per100":  "FG2A/100",
    "fg3a_rate_per100":  "FGA3/100",
    "fta_rate_per100":   "FTA/100",
    "usage_per100":      "USG%",
    "oefg":              "OeFG%",
    "otov":              "OTO%",
    "oorb":              "OORB%",
    "oftr":              "OFTR",
    "defg":              "DeFG%",
    "dtov":              "DTO%",
    "dorb":              "DORB%",
    "dftr":              "DFTR",
}

# Pre-compute leaderboard: latest non-null estimate per player per skill
print("Building leaderboard...")
leaderboard_rows = []
for player, grp in df.groupby("player"):
    row = {"player": player}
    for skill in SKILLS:
        s = grp[skill].dropna()
        row[skill] = round(float(s.iloc[-1]), 2) if len(s) else None
        last_date = grp.loc[grp[skill].notna(), "game_date"].max()
        row[f"{skill}_date"] = last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else None
    row["n_games"] = len(grp)
    leaderboard_rows.append(row)
LEADERBOARD_DF = pd.DataFrame(leaderboard_rows)

# Pre-compute percentiles (rank among players with non-null values)
for skill in SKILLS:
    LEADERBOARD_DF[f"{skill}_pct"] = (
        LEADERBOARD_DF[skill].rank(pct=True, na_option="keep") * 100
    ).round(1)
print("Leaderboard + percentiles ready.")

# ── Shared styles / nav ────────────────────────────────────────────────────────
NAV = """
<nav>
  <a href="/" class="nav-link" id="nav-rankings">Rankings</a>
  <a href="/explorer" class="nav-link" id="nav-explorer">Explorer</a>
  <button class="theme-toggle" id="themeToggle" title="Toggle light/dark">&#9790;</button>
</nav>
<script>
  (function() {
    if (localStorage.getItem('theme') === 'light') document.documentElement.classList.add('light-mode');
  })();
  function toggleTheme() {
    const light = document.documentElement.classList.toggle('light-mode');
    localStorage.setItem('theme', light ? 'light' : 'dark');
    document.getElementById('themeToggle').textContent = light ? '\u2600' : '\u263e';
    if (typeof onThemeChange === 'function') onThemeChange();
  }
  document.addEventListener('DOMContentLoaded', function() {
    const btn = document.getElementById('themeToggle');
    if (btn) {
      btn.textContent = document.documentElement.classList.contains('light-mode') ? '\u2600' : '\u263e';
      btn.onclick = toggleTheme;
    }
  });
</script>
"""

SHARED_CSS = """
:root {
  --bg:           #0d1117;
  --bg-card:      #161b22;
  --bg-card2:     #1c2128;
  --bg-hover:     #21262d;
  --border:       #30363d;
  --text:         #e6edf3;
  --text-muted:   #8b949e;
  --text-subtle:  #3d444d;
  --accent:       #58a6ff;
  --accent-rgb:   88,166,255;
  --highlight-bg: #1f3352;
}
html.light-mode {
  --bg:           #ffffff;
  --bg-card:      #f6f8fa;
  --bg-card2:     #eaeef2;
  --bg-hover:     #f3f4f6;
  --border:       #d0d7de;
  --text:         #1f2328;
  --text-muted:   #656d76;
  --text-subtle:  #9ca3af;
  --accent:       #0969da;
  --accent-rgb:   9,105,218;
  --highlight-bg: #dbeafe;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh;
}
header {
  background: var(--bg-card); border-bottom: 1px solid var(--border);
  padding: 14px 32px; display: flex; align-items: center; justify-content: space-between;
}
.logo { font-size: 20px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
.subtitle { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
nav { display: flex; gap: 6px; align-items: center; }
.nav-link {
  padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 500;
  color: var(--text-muted); text-decoration: none; border: 1px solid transparent; transition: all 0.15s;
}
.nav-link:hover { color: var(--text); background: var(--bg-hover); }
.nav-link.active { color: var(--text); background: var(--bg-hover); border-color: var(--border); }
.theme-toggle {
  background: none; border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-muted); cursor: pointer; padding: 5px 9px; font-size: 15px;
  transition: all 0.15s; line-height: 1;
}
.theme-toggle:hover { color: var(--text); background: var(--bg-hover); }
"""

# ── Explorer page ──────────────────────────────────────────────────────────────
EXPLORER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NBA Skills Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    """ + SHARED_CSS + """
    .controls {
      display: flex; gap: 16px; padding: 18px 32px;
      background: var(--bg-card); border-bottom: 1px solid var(--border);
      flex-wrap: wrap; align-items: flex-end;
    }
    .control-group { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    input[type="text"], select {
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 8px 12px; font-size: 14px; outline: none; transition: border-color 0.15s;
    }
    input[type="text"]:focus, select:focus { border-color: var(--accent); }
    input[type="text"] { width: 260px; }
    select { width: 200px; cursor: pointer; }
    .autocomplete-wrapper { position: relative; }
    #suggestions {
      position: absolute; top: 100%; left: 0; right: 0;
      background: var(--bg-card2); border: 1px solid var(--border); border-top: none;
      border-radius: 0 0 6px 6px; max-height: 240px; overflow-y: auto;
      z-index: 100; display: none;
    }
    #suggestions div { padding: 8px 12px; font-size: 14px; cursor: pointer; }
    #suggestions div:hover, #suggestions div.active { background: var(--bg-hover); color: var(--accent); }
    #loading { display: none; color: var(--text-muted); font-size: 12px; align-self: center; }
    .main {
      padding: 24px 32px;
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 24px;
      align-items: start;
    }
    .chart-card {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 24px 28px;
    }
    .chart-header { margin-bottom: 16px; }
    .chart-title { font-size: 17px; font-weight: 700; color: var(--text); }
    .chart-subtitle { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
    .stats-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    .stat-chip {
      background: var(--bg-hover); border: 1px solid var(--border); border-radius: 6px;
      padding: 8px 14px; font-size: 13px; min-width: 100px;
    }
    .stat-chip .val { font-weight: 700; color: var(--accent); font-size: 15px; }
    .stat-chip .lbl { color: var(--text-muted); font-size: 11px; margin-top: 2px; }
    .chart-outer { position: relative; }
    .chart-container { position: relative; height: 480px; }
    .legend { display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text-muted); }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; }
    .legend-line { width: 22px; height: 3px; border-radius: 2px; }
    .placeholder {
      position: absolute; inset: 0; display: flex;
      align-items: center; justify-content: center; color: var(--text-muted); font-size: 15px;
    }
    .lb-card {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
      overflow: hidden; display: flex; flex-direction: column;
    }
    .lb-header {
      padding: 16px 20px 12px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: baseline; justify-content: space-between; gap: 10px;
    }
    .lb-title { font-size: 14px; font-weight: 700; color: var(--text); }
    .lb-subtitle { font-size: 11px; color: var(--text-muted); }
    .lb-search {
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 7px 12px; font-size: 13px; outline: none; transition: border-color 0.15s;
      display: block; margin: 10px 20px 0; width: calc(100% - 40px);
    }
    .lb-search:focus { border-color: var(--accent); }
    .lb-table-wrap { overflow-y: auto; max-height: 560px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th {
      position: sticky; top: 0; background: var(--bg-card2);
      padding: 9px 14px; text-align: left; font-size: 11px;
      font-weight: 600; color: var(--text-muted); text-transform: uppercase;
      letter-spacing: 0.4px; border-bottom: 1px solid var(--border); white-space: nowrap;
      cursor: pointer; user-select: none;
    }
    thead th:hover { color: var(--text); }
    thead th.sorted { color: var(--accent); }
    thead th .sort-icon { margin-left: 4px; opacity: 0.5; }
    thead th.sorted .sort-icon { opacity: 1; }
    tbody tr { border-bottom: 1px solid var(--bg-hover); cursor: pointer; transition: background 0.1s; }
    tbody tr:hover { background: var(--bg-hover); }
    tbody tr.highlighted { background: var(--highlight-bg); }
    tbody tr.highlighted td.player-name { color: var(--accent); }
    tbody td { padding: 9px 14px; color: var(--text); white-space: nowrap; }
    tbody td.rank { color: var(--text-muted); font-size: 12px; width: 32px; }
    tbody td.player-name { font-weight: 500; max-width: 160px; overflow: hidden; text-overflow: ellipsis; }
    tbody td.skill-val { font-variant-numeric: tabular-nums; font-weight: 600; }
    tbody td.skill-val span {
      display: inline-block; padding: 2px 7px; border-radius: 4px;
      background: rgba(var(--accent-rgb),0.1); color: var(--accent);
    }
    .bar-cell { width: 80px; }
    .bar-bg { background: var(--bg-hover); border-radius: 3px; height: 6px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
    .lb-loading { padding: 32px; text-align: center; color: var(--text-muted); font-size: 13px; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="logo">NBA Skills Explorer</div>
      <div class="subtitle">Skill estimates per 100 possessions — dots = estimate, curve = LOESS smooth</div>
    </div>
    """ + NAV + """
  </header>

  <div class="controls">
    <div class="control-group">
      <label>Player</label>
      <div class="autocomplete-wrapper">
        <input type="text" id="playerInput" placeholder="Search player…" autocomplete="off" />
        <div id="suggestions"></div>
      </div>
    </div>
    <div class="control-group">
      <label>Skill</label>
      <select id="skillSelect">
        <optgroup label="Skills">
          {% for key, label in skills.items() %}
          <option value="{{ key }}">{{ label }}</option>
          {% endfor %}
        </optgroup>
        <optgroup label="SPM">
          {% for key, label in spm_skills.items() %}
          <option value="{{ key }}">{{ label }}</option>
          {% endfor %}
        </optgroup>
      </select>
    </div>
    <div id="loading">Loading…</div>
  </div>

  <div class="main">
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-title" id="chartTitle">Select a player to get started</div>
        <div class="chart-subtitle" id="chartSubtitle">Search above to explore a player's skill trajectory over time</div>
      </div>
      <div id="statsRow" class="stats-row" style="visibility:hidden"></div>
      <div class="chart-outer">
        <div class="chart-container">
          <canvas id="myChart"></canvas>
        </div>
        <div class="placeholder" id="placeholder">Search for a player above</div>
      </div>
      <div class="legend" id="legend" style="display:none">
        <div class="legend-item">
          <div class="legend-dot" style="background:var(--text-muted)"></div>
          Skill estimate (game)
        </div>
        <div class="legend-item">
          <div class="legend-line" style="background:var(--accent)"></div>
          LOESS smooth
        </div>
        <div class="legend-item">
          <div class="legend-line" style="background:transparent;border-top:2px dashed var(--text-muted);height:0;margin-top:4px"></div>
          League avg
        </div>
      </div>
    </div>

    <div class="lb-card">
      <div class="lb-header">
        <div>
          <div class="lb-title" id="lbTitle">Leaderboard</div>
          <div class="lb-subtitle" id="lbSubtitle">Latest skill estimate per player</div>
        </div>
      </div>
      <input class="lb-search" id="lbSearch" type="text" placeholder="Filter players…" />
      <div class="lb-table-wrap">
        <div class="lb-loading" id="lbLoading">Loading…</div>
        <table id="lbTable" style="display:none">
          <thead>
            <tr>
              <th data-col="rank">#<span class="sort-icon">↕</span></th>
              <th data-col="player">Player<span class="sort-icon">↕</span></th>
              <th data-col="value" class="sorted">Value<span class="sort-icon">↓</span></th>
              <th class="bar-cell"></th>
            </tr>
          </thead>
          <tbody id="lbBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const PLAYERS = {{ players | tojson }};
    const SKILL_LABELS = {{ skill_labels | tojson }};
    const SPM_SKILLS = new Set({{ spm_skills.keys() | list | tojson }});
    const FF_SKILLS  = new Set(['oefg','otov','oorb','oftr','defg','dtov','dorb','dftr']);
    let chart = null;
    let currentPlayer = null;
    let currentSkill = "spm";
    let lbData = [];
    let lbFiltered = [];
    let sortCol = "value";
    let sortAsc = false;

    function fmtVal(v, skill) {
      if (v == null) return "—";
      const decimals = (SPM_SKILLS.has(skill) || FF_SKILLS.has(skill)) ? 2 : 1;
      const sign = (SPM_SKILLS.has(skill) && v > 0) ? "+" : "";
      return sign + v.toFixed(decimals);
    }

    function getColors() {
      const light = document.documentElement.classList.contains('light-mode');
      return light ? {
        accent:    '#0969da',
        muted:     '#656d76',
        grid:      '#f3f4f6',
        border:    '#d0d7de',
        text:      '#1f2328',
        tooltipBg: '#f6f8fa',
        dot:       'rgba(101,109,118,0.55)',
      } : {
        accent:    '#58a6ff',
        muted:     '#8b949e',
        grid:      '#21262d',
        border:    '#30363d',
        text:      '#e6edf3',
        tooltipBg: '#21262d',
        dot:       'rgba(139,148,158,0.55)',
      };
    }

    function onThemeChange() {
      if (chart && currentPlayer) loadChart();
    }

    document.getElementById("nav-explorer").classList.add("active");

    const input = document.getElementById("playerInput");
    const suggestions = document.getElementById("suggestions");
    let activeIdx = -1;

    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      if (!q) { suggestions.style.display = "none"; return; }
      const matches = PLAYERS.filter(p => p.toLowerCase().includes(q)).slice(0, 20);
      if (!matches.length) { suggestions.style.display = "none"; return; }
      suggestions.innerHTML = matches.map(p => `<div data-name="${p}">${p}</div>`).join("");
      suggestions.style.display = "block";
      activeIdx = -1;
    });
    suggestions.addEventListener("click", e => {
      const d = e.target.closest("[data-name]");
      if (d) selectPlayer(d.dataset.name);
    });
    input.addEventListener("keydown", e => {
      const items = suggestions.querySelectorAll("div");
      if (!items.length) return;
      if (e.key === "ArrowDown") { activeIdx = Math.min(activeIdx+1, items.length-1); hlAc(items); e.preventDefault(); }
      else if (e.key === "ArrowUp") { activeIdx = Math.max(activeIdx-1, 0); hlAc(items); e.preventDefault(); }
      else if (e.key === "Enter" && activeIdx >= 0) { selectPlayer(items[activeIdx].dataset.name); e.preventDefault(); }
    });
    document.addEventListener("click", e => {
      if (!e.target.closest(".autocomplete-wrapper")) suggestions.style.display = "none";
    });
    function hlAc(items) { items.forEach((el, i) => el.classList.toggle("active", i === activeIdx)); }

    function selectPlayer(name) {
      currentPlayer = name;
      input.value = name;
      suggestions.style.display = "none";
      loadChart();
      highlightLeaderboard(name);
    }

    document.getElementById("skillSelect").addEventListener("change", () => {
      if (currentPlayer) loadChart();
      loadLeaderboard();
    });

    async function loadChart() {
      const skill = document.getElementById("skillSelect").value;
      document.getElementById("loading").style.display = "block";
      const res = await fetch(`/api/data?player=${encodeURIComponent(currentPlayer)}&skill=${skill}`);
      const data = await res.json();
      document.getElementById("loading").style.display = "none";
      renderChart(data, currentPlayer, skill);
    }

    function renderChart(data, player, skill) {
      const label = SKILL_LABELS[skill];
      document.getElementById("chartTitle").textContent = `${player} — ${label}`;
      document.getElementById("chartSubtitle").textContent = `${data.n_games} games · ${data.date_range}`;
      document.getElementById("placeholder").style.display = "none";
      document.getElementById("legend").style.display = "flex";

      const ests = data.estimate.filter(v => v !== null);
      if (ests.length) {
        const latest = data.estimate[data.estimate.length - 1] ?? ests[ests.length - 1];
        const avg = ests.reduce((a,b)=>a+b,0)/ests.length;
        const mn = Math.min(...ests);
        const mx = Math.max(...ests);
        document.getElementById("statsRow").style.visibility = "visible";
        document.getElementById("statsRow").innerHTML = `
          <div class="stat-chip"><div class="val">${fmtVal(latest, skill)}</div><div class="lbl">Latest estimate</div></div>
          <div class="stat-chip"><div class="val">${fmtVal(avg, skill)}</div><div class="lbl">Career avg</div></div>
          <div class="stat-chip"><div class="val">${fmtVal(mn, skill)} – ${fmtVal(mx, skill)}</div><div class="lbl">Range</div></div>
          <div class="stat-chip"><div class="val">${data.n_games}</div><div class="lbl">Games</div></div>
        `;
      } else {
        document.getElementById("statsRow").style.visibility = "hidden";
      }

      const c = getColors();
      if (chart) chart.destroy();
      const ctx = document.getElementById("myChart").getContext("2d");
      const n = data.dates.length;
      const leagueAvg = data.league_avg;
      const datasets = [
        {
          label: "Estimate",
          data: data.dates.map((d, i) =>
            data.estimate[i] != null ? { x: i, y: data.estimate[i], date: d } : null
          ).filter(Boolean),
          backgroundColor: c.dot,
          pointRadius: 4, pointHoverRadius: 6, order: 2,
        },
        {
          label: "LOESS",
          type: "line",
          data: data.loess.map((y, i) => y != null ? { x: i, y } : null).filter(Boolean),
          borderColor: c.accent, backgroundColor: "transparent",
          borderWidth: 2.5, pointRadius: 0, tension: 0.4, spanGaps: true, order: 1,
        },
      ];
      if (leagueAvg != null) {
        datasets.push({
          label: "League Avg",
          type: "line",
          data: [{ x: 0, y: leagueAvg }, { x: n - 1, y: leagueAvg }],
          borderColor: c.muted, backgroundColor: "transparent",
          borderWidth: 1.5, borderDash: [6, 4], pointRadius: 0, order: 3,
        });
      }
      chart = new Chart(ctx, {
        type: "scatter",
        data: { datasets },
        options: {
          responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
          interaction: { mode: "nearest", intersect: false, axis: "x" },
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: c.tooltipBg, borderColor: c.border, borderWidth: 1,
              titleColor: c.text, bodyColor: c.muted,
              callbacks: {
                title: items => { const idx = Math.round(items[0].raw.x); return data.dates[idx] ?? ""; },
                label: item => {
                  if (item.dataset.label === "Estimate") return `Estimate: ${item.raw.y.toFixed(2)}`;
                  if (item.dataset.label === "LOESS") return `LOESS: ${item.raw.y.toFixed(2)}`;
                  if (item.dataset.label === "League Avg") return `League Avg: ${item.raw.y.toFixed(2)}`;
                  return null;
                }
              }
            }
          },
          scales: {
            x: {
              type: "linear",
              title: { display: true, text: "Game #", color: c.muted, font: { size: 12 } },
              ticks: { color: c.muted, font: { size: 11 }, maxTicksLimit: 10, callback: v => Math.round(v) },
              grid: { color: c.grid }, border: { color: c.border },
            },
            y: {
              title: { display: true, text: label, color: c.muted, font: { size: 12 } },
              ticks: { color: c.muted, font: { size: 11 } },
              grid: { color: c.grid }, border: { color: c.border },
            }
          }
        }
      });
    }

    async function loadLeaderboard() {
      currentSkill = document.getElementById("skillSelect").value;
      const skill = currentSkill;
      document.getElementById("lbLoading").style.display = "block";
      document.getElementById("lbTable").style.display = "none";
      document.getElementById("lbTitle").textContent = `${SKILL_LABELS[skill] || skill} Leaderboard`;

      const res = await fetch(`/api/leaderboard?skill=${skill}`);
      lbData = await res.json();
      document.getElementById("lbLoading").style.display = "none";
      document.getElementById("lbTable").style.display = "table";
      applyLbFilter();
    }

    function applyLbFilter() {
      const q = document.getElementById("lbSearch").value.trim().toLowerCase();
      lbFiltered = q ? lbData.filter(r => r.player.toLowerCase().includes(q)) : [...lbData];
      sortAndRender();
    }

    function sortAndRender() {
      lbFiltered.sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (va == null) return 1; if (vb == null) return -1;
        return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
      });
      renderLbRows();
    }

    function renderLbRows() {
      const vals = lbFiltered.map(r => r.value).filter(v => v != null);
      const minVal = vals.length ? Math.min(...vals) : 0;
      const maxVal = vals.length ? Math.max(...vals) : 1;
      const range = maxVal - minVal || 1;
      const body = document.getElementById("lbBody");
      body.innerHTML = lbFiltered.map((row, i) => {
        const barPct = row.value != null ? Math.round(((row.value - minVal) / range) * 100) : 0;
        const isHL = row.player === currentPlayer;
        return `<tr class="${isHL ? "highlighted" : ""}" data-player="${row.player}">
          <td class="rank">${i + 1}</td>
          <td class="player-name" title="${row.player}">${row.player}</td>
          <td class="skill-val"><span>${fmtVal(row.value, currentSkill)}</span></td>
          <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${barPct}%"></div></div></td>
        </tr>`;
      }).join("");

      body.querySelectorAll("tr[data-player]").forEach(tr => {
        tr.addEventListener("click", () => selectPlayer(tr.dataset.player));
      });

      const hlRow = body.querySelector("tr.highlighted");
      if (hlRow) hlRow.scrollIntoView({ block: "nearest" });
    }

    function highlightLeaderboard(name) {
      document.querySelectorAll("#lbBody tr").forEach(tr => {
        tr.classList.toggle("highlighted", tr.dataset.player === name);
        if (tr.dataset.player === name) {
          tr.querySelector("td.player-name").style.color = "var(--accent)";
          tr.scrollIntoView({ block: "nearest" });
        } else {
          tr.querySelector("td.player-name").style.color = "";
        }
      });
    }

    document.querySelectorAll("thead th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (sortCol === col) sortAsc = !sortAsc;
        else { sortCol = col; sortAsc = col === "player"; }
        document.querySelectorAll("thead th").forEach(h => {
          h.classList.remove("sorted");
          h.querySelector(".sort-icon").textContent = "↕";
        });
        th.classList.add("sorted");
        th.querySelector(".sort-icon").textContent = sortAsc ? "↑" : "↓";
        sortAndRender();
      });
    });

    document.getElementById("lbSearch").addEventListener("input", applyLbFilter);
    document.getElementById("skillSelect").value = "spm";
    loadLeaderboard().then(() => selectPlayer("Nikola Jokic"));

    // Keep leaderboard the same height as the chart card
    (function() {
      const chartCard = document.querySelector('.chart-card');
      const lbCard    = document.querySelector('.lb-card');
      const lbWrap    = document.querySelector('.lb-table-wrap');
      function sync() {
        const overhead = lbCard.offsetHeight - lbWrap.offsetHeight;
        lbWrap.style.maxHeight = Math.max(200, chartCard.offsetHeight - overhead) + 'px';
      }
      new ResizeObserver(sync).observe(chartCard);
    })();
  </script>
</body>
</html>
"""

# ── Rankings page ──────────────────────────────────────────────────────────────
RANKINGS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NBA Skills Rankings</title>
  <style>
    """ + SHARED_CSS + """
    .page-body { padding: 24px 32px; }
    .toolbar {
      display: flex; gap: 12px; align-items: center; margin-bottom: 18px; flex-wrap: wrap;
    }
    .toolbar input {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 8px 14px; font-size: 14px; outline: none;
      transition: border-color 0.15s; width: 260px;
    }
    .toolbar input:focus { border-color: var(--accent); }
    .toolbar .count { font-size: 12px; color: var(--text-muted); margin-left: 4px; }
    .table-wrap {
      overflow-x: auto; overflow-y: auto; max-height: calc(100vh - 200px);
      border: 1px solid var(--border); border-radius: 10px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }
    thead tr { background: var(--bg-card2); }
    thead th {
      padding: 10px 14px; text-align: center; font-size: 11px;
      font-weight: 600; color: var(--text-muted); text-transform: uppercase;
      letter-spacing: 0.4px; border-bottom: 1px solid var(--border);
      cursor: pointer; user-select: none; position: sticky; top: 0;
      background: var(--bg-card2);
    }
    thead th:hover { color: var(--text); }
    thead th.sorted { color: var(--accent); }
    thead th .sort-icon { margin-left: 3px; opacity: 0.5; }
    thead th.sorted .sort-icon { opacity: 1; }
    thead th.sticky-col { left: 0; z-index: 2; }
    tbody tr { border-bottom: 1px solid var(--bg-hover); transition: background 0.1s; }
    tbody tr:hover { background: var(--bg-card); }
    tbody td { padding: 8px 14px; color: var(--text); vertical-align: middle; }
    td.sticky-col {
      position: sticky; left: 0; background: var(--bg); z-index: 1;
      font-weight: 500; min-width: 160px; max-width: 200px;
      overflow: hidden; text-overflow: ellipsis;
    }
    tbody tr:hover td.sticky-col { background: var(--bg-card); }
    td.rank-col { color: var(--text-muted); font-size: 12px; width: 40px; text-align: right; padding-right: 8px; }
    td.games-col { color: var(--text-muted); font-size: 12px; text-align: center; }
    td.stat-cell { text-align: center; min-width: 88px; }
    .cell-inner { display: flex; flex-direction: column; align-items: center; gap: 2px; }
    .cell-val { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--text); font-size: 13px; }
    .cell-pct {
      font-size: 10px; font-weight: 600; padding: 1px 5px; border-radius: 3px;
      color: var(--text); letter-spacing: 0.2px;
    }
    .cell-null { color: var(--text-subtle); font-size: 12px; text-align: center; }
    .loading { padding: 60px; text-align: center; color: var(--text-muted); font-size: 14px; }
    #col-tooltip {
      position: fixed; z-index: 9999; pointer-events: none;
      background: var(--bg-card2); border: 1px solid var(--border); border-radius: 7px;
      padding: 9px 13px; max-width: 280px; box-shadow: 0 4px 16px rgba(0,0,0,0.35);
      display: none;
    }
    #col-tooltip .tt-abbr { font-size: 12px; font-weight: 700; color: var(--accent); margin-bottom: 4px; }
    #col-tooltip .tt-desc { font-size: 12px; color: var(--text-muted); line-height: 1.45; white-space: normal; }
    .glossary {
      margin-top: 28px; border: 1px solid var(--border); border-radius: 10px;
      background: var(--bg-card); padding: 20px 28px;
    }
    .glossary h3 {
      font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;
      letter-spacing: 0.5px; margin-bottom: 16px;
    }
    .glossary-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 0 48px;
    }
    .glossary-section { margin-bottom: 16px; }
    .glossary-section h4 {
      font-size: 11px; font-weight: 700; color: var(--accent); text-transform: uppercase;
      letter-spacing: 0.4px; margin-bottom: 8px;
    }
    .glossary-row {
      display: flex; gap: 10px; padding: 5px 0;
      border-bottom: 1px solid var(--bg-hover); align-items: baseline;
    }
    .glossary-row:last-child { border-bottom: none; }
    .glossary-abbr {
      font-size: 12px; font-weight: 700; color: var(--text); min-width: 80px;
      font-variant-numeric: tabular-nums;
    }
    .glossary-desc { font-size: 12px; color: var(--text-muted); line-height: 1.4; }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="logo">NBA Skills Rankings</div>
      <div class="subtitle">Latest skill estimate + percentile rank for every player</div>
    </div>
    """ + NAV + """
  </header>

  <div id="col-tooltip"><div class="tt-abbr" id="tt-abbr"></div><div class="tt-desc" id="tt-desc"></div></div>

  <div class="page-body">
    <div class="toolbar">
      <input type="text" id="filterInput" placeholder="Filter players…" />
      <span class="count" id="rowCount"></span>
    </div>
    <div class="table-wrap" id="tableWrap">
      <div class="loading" id="loadingMsg">Loading…</div>
    </div>

    <div class="glossary">
      <h3>Column Glossary</h3>
      <div class="glossary-grid">

        <div class="glossary-section">
          <h4>Box Score Skills</h4>
          <div class="glossary-row"><span class="glossary-abbr">PTS/100</span><span class="glossary-desc">Points scored per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">AST/100</span><span class="glossary-desc">Assists per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">OREB/100</span><span class="glossary-desc">Offensive rebounds per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">DREB/100</span><span class="glossary-desc">Defensive rebounds per 100 defensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">STL/100</span><span class="glossary-desc">Steals per 100 defensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">BLK/100</span><span class="glossary-desc">Blocks per 100 defensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">TOV/100</span><span class="glossary-desc">Turnovers per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">PF/100</span><span class="glossary-desc">Personal fouls per 100 defensive possessions</span></div>
        </div>

        <div class="glossary-section">
          <h4>Shooting Skills</h4>
          <div class="glossary-row"><span class="glossary-abbr">FG2%</span><span class="glossary-desc">2-point field goal percentage</span></div>
          <div class="glossary-row"><span class="glossary-abbr">FG3%</span><span class="glossary-desc">3-point field goal percentage</span></div>
          <div class="glossary-row"><span class="glossary-abbr">FT%</span><span class="glossary-desc">Free throw percentage</span></div>
          <div class="glossary-row"><span class="glossary-abbr">FG2A/100</span><span class="glossary-desc">2-point attempts per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">FG3A/100</span><span class="glossary-desc">3-point attempts per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">FTA/100</span><span class="glossary-desc">Free throw attempts per 100 offensive possessions</span></div>
          <div class="glossary-row"><span class="glossary-abbr">USG%</span><span class="glossary-desc">Usage rate — share of team plays used (FGA + 0.44×FTA + TOV) per offensive possession</span></div>
        </div>

        <div class="glossary-section">
          <h4>Four Factors — Offense</h4>
          <div class="glossary-row"><span class="glossary-abbr">OeFG%</span><span class="glossary-desc">Team eFG% impact when on floor offensively</span></div>
          <div class="glossary-row"><span class="glossary-abbr">OTO%</span><span class="glossary-desc">Reduction in team turnover rate on offense (negated, so higher = fewer TOs)</span></div>
          <div class="glossary-row"><span class="glossary-abbr">OORB%</span><span class="glossary-desc">Team offensive rebound rate impact when on floor offensively</span></div>
          <div class="glossary-row"><span class="glossary-abbr">OFTR</span><span class="glossary-desc">Team free throw attempts per possession impact on offense</span></div>
        </div>

        <div class="glossary-section">
          <h4>Four Factors — Defense</h4>
          <div class="glossary-row"><span class="glossary-abbr">DeFG%</span><span class="glossary-desc">Suppression of opponent eFG% when on floor defensively (higher = better impact)</span></div>
          <div class="glossary-row"><span class="glossary-abbr">DTO%</span><span class="glossary-desc">Increase in opponent turnover rate when on floor defensively (higher = more forced TOs)</span></div>
          <div class="glossary-row"><span class="glossary-abbr">DORB%</span><span class="glossary-desc">Reduction in opponent offensive rebound rate when on floor defensively (higher = better)</span></div>
          <div class="glossary-row"><span class="glossary-abbr">DFTR</span><span class="glossary-desc">Reduction in opponent FTA per possession when on floor defensively (higher = fewer FTA drawn)</span></div>
        </div>

        <div class="glossary-section">
          <h4>SPM</h4>
          <div class="glossary-row"><span class="glossary-abbr">OSPM</span><span class="glossary-desc">Offensive Statistical Plus-Minus — estimated offensive impact in points per 100 possessions above average, modeled from box score and four-factor skills via ridge regression fitted to O-RAPM</span></div>
          <div class="glossary-row"><span class="glossary-abbr">DSPM</span><span class="glossary-desc">Defensive Statistical Plus-Minus — same methodology fitted to D-RAPM</span></div>
          <div class="glossary-row"><span class="glossary-abbr">SPM</span><span class="glossary-desc">Total SPM = OSPM + DSPM</span></div>
        </div>

        <div class="glossary-section">
          <h4>Notes</h4>
          <div class="glossary-row"><span class="glossary-abbr">Estimates</span><span class="glossary-desc">All skills are rolling ridge regression estimates — each game's value reflects all prior games, with recent games weighted more heavily</span></div>
          <div class="glossary-row"><span class="glossary-abbr">Percentiles</span><span class="glossary-desc">Colored badges show rank among all players in the dataset. Green = top, red = bottom. TOV/100 and PF/100 are flipped (lower is better)</span></div>
          <div class="glossary-row"><span class="glossary-abbr">Four Factors</span><span class="glossary-desc">Estimated via APM — players separated from teammates using a +1/−1 lineup design matrix, then regularized with calendar-day decay weights</span></div>
        </div>

      </div>
    </div>
  </div>

  <script>
    const SKILLS = {{ skills | tojson }};
    const SKILL_ABBR = {{ skill_abbr | tojson }};

    let allRows = [];
    let sortKey = "pts_per100";
    let sortAsc = false;

    document.getElementById("nav-rankings").classList.add("active");

    const FLIP_SKILLS  = new Set(['tov_per100', 'pf_per100', 'fg2a_rate_per100']);
    const FF_SKILLS_R  = new Set(['oefg','otov','oorb','oftr','defg','dtov','dorb','dftr']);
    const SPM_SKILLS_R = new Set(['ospm','dspm','spm']);
    const SPM_COLS = ['ospm', 'dspm', 'spm'];

    const COL_TOOLTIPS = {
      pts_per100:       ["PTS/100",   "Points scored per 100 offensive possessions"],
      ast_per100:       ["AST/100",   "Assists per 100 offensive possessions"],
      oreb_per100:      ["OREB/100",  "Offensive rebounds per 100 offensive possessions"],
      dreb_per100:      ["DREB/100",  "Defensive rebounds per 100 defensive possessions"],
      stl_per100:       ["STL/100",   "Steals per 100 defensive possessions"],
      blk_per100:       ["BLK/100",   "Blocks per 100 defensive possessions"],
      tov_per100:       ["TOV/100",   "Turnovers per 100 offensive possessions"],
      pf_per100:        ["PF/100",    "Personal fouls per 100 defensive possessions"],
      fg2_pct:          ["FG2%",      "2-point field goal percentage"],
      fg3_pct:          ["FG3%",      "3-point field goal percentage"],
      ft_pct:           ["FT%",       "Free throw percentage"],
      fg2a_rate_per100: ["FG2A/100",  "2-point attempts per 100 offensive possessions"],
      fg3a_rate_per100: ["FG3A/100",  "3-point attempts per 100 offensive possessions"],
      fta_rate_per100:  ["FTA/100",   "Free throw attempts per 100 offensive possessions"],
      usage_per100:     ["USG%",      "Usage rate — share of team plays used (FGA + 0.44×FTA + TOV) per offensive possession"],
      oefg:             ["OeFG%",     "Team eFG% impact when on floor offensively"],
      otov:             ["OTO%",      "Reduction in team turnover rate on offense (negated, so higher = fewer TOs)"],
      oorb:             ["OORB%",     "Team offensive rebound rate impact when on floor offensively"],
      oftr:             ["OFTR",      "Team free throw attempts per possession impact on offense"],
      defg:             ["DeFG%",     "Suppression of opponent eFG% when on floor defensively (higher = better impact)"],
      dtov:             ["DTO%",      "Increase in opponent turnover rate when on floor defensively (higher = more forced TOs)"],
      dorb:             ["DORB%",     "Reduction in opponent offensive rebound rate when on floor defensively (higher = better)"],
      dftr:             ["DFTR",      "Reduction in opponent FTA per possession when on floor defensively (higher = fewer FTA drawn)"],
      ospm:             ["OSPM",      "Offensive Statistical Plus-Minus — estimated offensive impact in points per 100 possessions above average, modeled from box score and four-factor skills via ridge regression fitted to O-RAPM"],
      dspm:             ["DSPM",      "Defensive Statistical Plus-Minus — same methodology fitted to D-RAPM"],
      spm:              ["SPM",       "Total SPM = OSPM + DSPM"],
      n_games:          ["G",         "Number of games in the dataset for this player"],
    };

    const ttEl   = document.getElementById("col-tooltip");
    const ttAbbr = document.getElementById("tt-abbr");
    const ttDesc = document.getElementById("tt-desc");

    function showTooltip(e, key) {
      const tip = COL_TOOLTIPS[key];
      if (!tip) return;
      ttAbbr.textContent = tip[0];
      ttDesc.textContent = tip[1];
      ttEl.style.display = "block";
      positionTooltip(e);
    }
    function positionTooltip(e) {
      const pad = 12;
      let x = e.clientX + pad, y = e.clientY + pad;
      const w = ttEl.offsetWidth, h = ttEl.offsetHeight;
      if (x + w > window.innerWidth  - pad) x = e.clientX - w - pad;
      if (y + h > window.innerHeight - pad) y = e.clientY - h - pad;
      ttEl.style.left = x + "px";
      ttEl.style.top  = y + "px";
    }
    document.addEventListener("mousemove", e => {
      if (ttEl.style.display !== "none") positionTooltip(e);
    });

    function pctColor(pct, flip = false) {
      if (pct == null) return 'transparent';
      const p = flip ? 100 - pct : pct;
      const hue = p * 1.2; // 0 → red (0°), 100 → green (120°)
      return `hsla(${hue.toFixed(1)}, 60%, 48%, 0.35)`;
    }

    function onThemeChange() {
      applyFilterAndSort();
    }

    function pctLabel(pct) {
      if (pct == null) return "—";
      return Math.round(pct) + "%";
    }

    function buildTable(rows) {
      const wrap = document.getElementById("tableWrap");

      const SPM_LABEL = {ospm: "OSPM", dspm: "DSPM", spm: "SPM"};
      const SPM_COLOR = {ospm: "#79c0ff", dspm: "#79c0ff", spm: "#f0883e"};

      let headerCells = `
        <th class="sticky-col" data-key="player" style="min-width:160px;text-align:left">
          Player<span class="sort-icon">↕</span>
        </th>
        <th data-key="n_games" data-tip="n_games" style="text-align:center">
          G<span class="sort-icon">↕</span>
        </th>
      `;
      for (const sk of SKILLS) {
        const active = sk === sortKey ? ' class="sorted"' : '';
        headerCells += `<th${active} data-key="${sk}" data-tip="${sk}" class="stat-cell">
          ${SKILL_ABBR[sk]}<span class="sort-icon">${sk === sortKey ? (sortAsc ? "↑" : "↓") : "↕"}</span>
        </th>`;
      }
      for (const sc of SPM_COLS) {
        const active = sc === sortKey ? ' class="sorted"' : '';
        headerCells += `<th${active} data-key="${sc}" data-tip="${sc}" class="stat-cell" style="border-left:1px solid var(--border)">
          ${SPM_LABEL[sc]}<span class="sort-icon">${sc === sortKey ? (sortAsc ? "↑" : "↓") : "↕"}</span>
        </th>`;
      }

      const bodyRows = rows.map((r, i) => {
        let cells = `
          <td class="rank-col">${i + 1}</td>
          <td class="sticky-col" title="${r.player}">${r.player}</td>
          <td class="games-col">${r.n_games}</td>
        `;
        for (const sk of SKILLS) {
          const val = r[sk];
          const pct = r[sk + "_pct"];
          if (val == null) {
            cells += `<td class="stat-cell"><span class="cell-null">—</span></td>`;
          } else {
            const decimals = FF_SKILLS_R.has(sk) ? 2 : 1;
            const sign = FF_SKILLS_R.has(sk) && val > 0 ? "+" : "";
            cells += `<td class="stat-cell">
              <div class="cell-inner">
                <span class="cell-val">${sign}${val.toFixed(decimals)}</span>
                <span class="cell-pct" style="background:${pctColor(pct, FLIP_SKILLS.has(sk))}">${pctLabel(pct)}</span>
              </div>
            </td>`;
          }
        }
        for (const sc of SPM_COLS) {
          const val = r[sc];
          const pct = r[sc + "_pct"];
          if (val == null) {
            cells += `<td class="stat-cell" style="border-left:${sc==="ospm"?"1px solid var(--border)":"none"}"><span class="cell-null">—</span></td>`;
          } else {
            const sign = val > 0 ? "+" : "";
            cells += `<td class="stat-cell" style="border-left:${sc==="ospm"?"1px solid var(--border)":"none"}">
              <div class="cell-inner">
                <span class="cell-val" style="color:${SPM_COLOR[sc]}">${sign}${val.toFixed(1)}</span>
                <span class="cell-pct" style="background:${pctColor(pct)}">${pctLabel(pct)}</span>
              </div>
            </td>`;
          }
        }
        return `<tr>${cells}</tr>`;
      }).join("");

      wrap.innerHTML = `
        <table>
          <thead><tr><th class="rank-col sticky-col" style="left:0;z-index:3"></th>${headerCells}</tr></thead>
          <tbody id="rankBody">${bodyRows}</tbody>
        </table>
      `;

      // attach sort listeners
      wrap.querySelectorAll("thead th[data-key]").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.key;
          if (sortKey === key) sortAsc = !sortAsc;
          else { sortKey = key; sortAsc = key === "player"; }
          applyFilterAndSort();
        });
      });

      // attach tooltip listeners
      wrap.querySelectorAll("thead th[data-tip]").forEach(th => {
        th.addEventListener("mouseenter", e => showTooltip(e, th.dataset.tip));
        th.addEventListener("mouseleave", () => { ttEl.style.display = "none"; });
      });
    }

    function applyFilterAndSort() {
      const q = document.getElementById("filterInput").value.trim().toLowerCase();
      let rows = q ? allRows.filter(r => r.player.toLowerCase().includes(q)) : [...allRows];

      rows.sort((a, b) => {
        let va = a[sortKey], vb = b[sortKey];
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortAsc ? va - vb : vb - va;
      });

      document.getElementById("rowCount").textContent = `${rows.length} player${rows.length !== 1 ? "s" : ""}`;
      buildTable(rows);
    }

    async function init() {
      const res = await fetch("/api/rankings");
      allRows = await res.json();
      document.getElementById("loadingMsg")?.remove();
      applyFilterAndSort();
    }

    document.getElementById("filterInput").addEventListener("input", applyFilterAndSort);
    init();
  </script>
</body>
</html>
"""

# ── SPM page ───────────────────────────────────────────────────────────────────
SPM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NBA SPM</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    """ + SHARED_CSS + """
    .controls {
      display: flex; gap: 16px; padding: 18px 32px;
      background: var(--bg-card); border-bottom: 1px solid var(--border);
      flex-wrap: wrap; align-items: flex-end;
    }
    .control-group { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    input[type="text"] {
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 8px 12px; font-size: 14px; outline: none;
      transition: border-color 0.15s; width: 260px;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    .autocomplete-wrapper { position: relative; }
    #suggestions {
      position: absolute; top: 100%; left: 0; right: 0;
      background: var(--bg-card2); border: 1px solid var(--border); border-top: none;
      border-radius: 0 0 6px 6px; max-height: 240px; overflow-y: auto;
      z-index: 100; display: none;
    }
    #suggestions div { padding: 8px 12px; font-size: 14px; cursor: pointer; }
    #suggestions div:hover, #suggestions div.active { background: var(--bg-hover); color: var(--accent); }
    #loading { display: none; color: var(--text-muted); font-size: 12px; align-self: center; }
    .main {
      padding: 24px 32px;
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 24px;
      align-items: start;
    }
    .chart-card {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 24px 28px;
    }
    .chart-header { margin-bottom: 16px; }
    .chart-title { font-size: 17px; font-weight: 700; color: var(--text); }
    .chart-subtitle { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
    .stats-row { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    .stat-chip {
      background: var(--bg-hover); border: 1px solid var(--border); border-radius: 6px;
      padding: 8px 14px; font-size: 13px; min-width: 90px;
    }
    .stat-chip .val { font-weight: 700; font-size: 15px; }
    .stat-chip .lbl { color: var(--text-muted); font-size: 11px; margin-top: 2px; }
    .chip-spm  .val { color: var(--accent); }
    .chip-ospm .val { color: #f0883e; }
    .chip-dspm .val { color: #79c0ff; }
    .chart-container { position: relative; height: 460px; }
    .placeholder {
      position: absolute; inset: 0; display: flex;
      align-items: center; justify-content: center; color: var(--text-muted); font-size: 15px;
    }
    .legend { display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text-muted); }
    .legend-line { width: 22px; height: 3px; border-radius: 2px; }
    .lb-card {
      background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
      overflow: hidden; display: flex; flex-direction: column;
    }
    .lb-header { padding: 16px 20px 12px; border-bottom: 1px solid var(--border); }
    .lb-title { font-size: 14px; font-weight: 700; color: var(--text); }
    .lb-subtitle { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .lb-search {
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      color: var(--text); padding: 7px 12px; font-size: 13px; outline: none;
      transition: border-color 0.15s; display: block; margin: 10px 20px 0; width: calc(100% - 40px);
    }
    .lb-search:focus { border-color: var(--accent); }
    .lb-table-wrap { overflow-y: auto; max-height: 560px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th {
      position: sticky; top: 0; background: var(--bg-card2);
      padding: 9px 10px; text-align: right; font-size: 11px;
      font-weight: 600; color: var(--text-muted); text-transform: uppercase;
      letter-spacing: 0.4px; border-bottom: 1px solid var(--border); white-space: nowrap;
      cursor: pointer; user-select: none;
    }
    thead th:first-child { text-align: left; }
    thead th:hover { color: var(--text); }
    thead th.sorted { color: var(--accent); }
    thead th .sort-icon { margin-left: 3px; opacity: 0.5; }
    thead th.sorted .sort-icon { opacity: 1; }
    tbody tr { border-bottom: 1px solid var(--bg-hover); cursor: pointer; transition: background 0.1s; }
    tbody tr:hover { background: var(--bg-hover); }
    tbody tr.highlighted { background: var(--highlight-bg); }
    tbody td { padding: 8px 10px; white-space: nowrap; text-align: right; }
    tbody td:first-child { text-align: left; font-weight: 500; max-width: 150px; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
    .col-spm  { color: var(--accent); font-weight: 700; }
    .col-ospm { color: #f0883e; font-weight: 600; }
    .col-dspm { color: #79c0ff; font-weight: 600; }
    .col-rapm { color: var(--text-muted); }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="logo">NBA SPM</div>
      <div class="subtitle">Statistical Plus-Minus — box-stat model of player impact (pts/100 above avg)</div>
    </div>
    """ + NAV + """
  </header>

  <div class="controls">
    <div class="control-group">
      <label>Player</label>
      <div class="autocomplete-wrapper">
        <input type="text" id="playerInput" placeholder="Search player…" autocomplete="off" />
        <div id="suggestions"></div>
      </div>
    </div>
    <div id="loading">Loading…</div>
  </div>

  <div class="main">
    <div class="chart-card">
      <div class="chart-header">
        <div class="chart-title" id="chartTitle">Select a player to get started</div>
        <div class="chart-subtitle" id="chartSubtitle">SPM trajectory game by game</div>
      </div>
      <div id="statsRow" class="stats-row" style="visibility:hidden"></div>
      <div style="position:relative">
        <div class="chart-container">
          <canvas id="myChart"></canvas>
        </div>
        <div class="placeholder" id="placeholder">Search for a player above</div>
      </div>
      <div class="legend" id="legend" style="display:none">
        <div class="legend-item"><div class="legend-line" style="background:#f0883e"></div>O-SPM</div>
        <div class="legend-item"><div class="legend-line" style="background:#79c0ff"></div>D-SPM</div>
        <div class="legend-item"><div class="legend-line" style="background:var(--accent)"></div>SPM</div>
      </div>
    </div>

    <div class="lb-card">
      <div class="lb-header">
        <div class="lb-title">Career SPM Leaderboard</div>
        <div class="lb-subtitle">2017–26 · click to view trajectory</div>
      </div>
      <input class="lb-search" id="lbSearch" type="text" placeholder="Filter players…" />
      <div class="lb-table-wrap">
        <table>
          <thead>
            <tr>
              <th data-col="player" style="text-align:left">Player<span class="sort-icon">↕</span></th>
              <th data-col="ospm">OSPM<span class="sort-icon">↕</span></th>
              <th data-col="dspm">DSPM<span class="sort-icon">↕</span></th>
              <th data-col="spm" class="sorted">SPM<span class="sort-icon">↓</span></th>
              <th data-col="orapm">ORAPM<span class="sort-icon">↕</span></th>
              <th data-col="drapm">DRAPM<span class="sort-icon">↕</span></th>
              <th data-col="rapm">RAPM<span class="sort-icon">↕</span></th>
            </tr>
          </thead>
          <tbody id="lbBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const PLAYERS = {{ players | tojson }};
    const CAREER  = {{ career | tojson }};
    let chart = null;
    let currentPlayer = null;
    let lbData = [...CAREER];
    let lbFiltered = [...CAREER];
    let sortCol = "spm";
    let sortAsc = false;

    function getColors() {
      const light = document.documentElement.classList.contains('light-mode');
      return light ? {
        accent: '#0969da', grid: '#f3f4f6', border: '#d0d7de',
        text: '#1f2328', muted: '#656d76', tooltipBg: '#f6f8fa',
      } : {
        accent: '#58a6ff', grid: '#21262d', border: '#30363d',
        text: '#e6edf3', muted: '#8b949e', tooltipBg: '#21262d',
      };
    }
    function onThemeChange() { if (chart && currentPlayer) loadChart(); }

    document.getElementById("nav-spm").classList.add("active");

    const input = document.getElementById("playerInput");
    const suggestions = document.getElementById("suggestions");
    let activeIdx = -1;

    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      if (!q) { suggestions.style.display = "none"; return; }
      const matches = PLAYERS.filter(p => p.toLowerCase().includes(q)).slice(0, 20);
      if (!matches.length) { suggestions.style.display = "none"; return; }
      suggestions.innerHTML = matches.map(p => `<div data-name="${p}">${p}</div>`).join("");
      suggestions.style.display = "block";
      activeIdx = -1;
    });
    suggestions.addEventListener("click", e => {
      const d = e.target.closest("[data-name]");
      if (d) selectPlayer(d.dataset.name);
    });
    input.addEventListener("keydown", e => {
      const items = suggestions.querySelectorAll("div");
      if (!items.length) return;
      if (e.key === "ArrowDown") { activeIdx = Math.min(activeIdx+1, items.length-1); hlAc(items); e.preventDefault(); }
      else if (e.key === "ArrowUp") { activeIdx = Math.max(activeIdx-1, 0); hlAc(items); e.preventDefault(); }
      else if (e.key === "Enter" && activeIdx >= 0) { selectPlayer(items[activeIdx].dataset.name); e.preventDefault(); }
    });
    document.addEventListener("click", e => {
      if (!e.target.closest(".autocomplete-wrapper")) suggestions.style.display = "none";
    });
    function hlAc(items) { items.forEach((el, i) => el.classList.toggle("active", i === activeIdx)); }

    function selectPlayer(name) {
      currentPlayer = name;
      input.value = name;
      suggestions.style.display = "none";
      loadChart();
      highlightLeaderboard(name);
    }

    async function loadChart() {
      document.getElementById("loading").style.display = "block";
      const res = await fetch(`/api/spm_data?player=${encodeURIComponent(currentPlayer)}`);
      const data = await res.json();
      document.getElementById("loading").style.display = "none";
      if (data.error) return;
      renderChart(data);
    }

    function fmt(v) { return v != null ? (v > 0 ? "+" : "") + v.toFixed(1) : "—"; }

    function renderChart(data) {
      document.getElementById("chartTitle").textContent = currentPlayer + " — SPM";
      document.getElementById("chartSubtitle").textContent = `${data.n_games} games · ${data.date_range}`;
      document.getElementById("placeholder").style.display = "none";
      document.getElementById("legend").style.display = "flex";

      const car = CAREER.find(r => r.player === currentPlayer);
      const latestSpm  = data.spm[data.spm.length - 1];
      const latestOspm = data.ospm[data.ospm.length - 1];
      const latestDspm = data.dspm[data.dspm.length - 1];
      document.getElementById("statsRow").style.visibility = "visible";
      document.getElementById("statsRow").innerHTML = `
        <div class="stat-chip chip-ospm"><div class="val">${fmt(latestOspm)}</div><div class="lbl">O-SPM (latest)</div></div>
        <div class="stat-chip chip-dspm"><div class="val">${fmt(latestDspm)}</div><div class="lbl">D-SPM (latest)</div></div>
        <div class="stat-chip chip-spm"><div class="val">${fmt(latestSpm)}</div><div class="lbl">SPM (latest)</div></div>
        ${car ? `<div class="stat-chip"><div class="val" style="color:var(--text-muted)">${fmt(car.rapm)}</div><div class="lbl">Career RAPM</div></div>` : ""}
      `;

      const c = getColors();
      const N = data.dates.length;
      const xs = Array.from({length: N}, (_, i) => i);

      function makeLine(smoothArr, color, label) {
        return {
          label, type: "line",
          data: smoothArr.map((y, i) => y != null ? {x: i, y} : null).filter(Boolean),
          borderColor: color, backgroundColor: "transparent",
          borderWidth: 2.5, pointRadius: 0, tension: 0.3, spanGaps: true,
        };
      }

      if (chart) chart.destroy();
      chart = new Chart(document.getElementById("myChart").getContext("2d"), {
        type: "scatter",
        data: { datasets: [
          makeLine(data.ospm_sm, "#f0883e", "O-SPM"),
          makeLine(data.dspm_sm, "#79c0ff", "D-SPM"),
          makeLine(data.spm_sm,  c.accent,  "SPM"),
        ]},
        options: {
          responsive: true, maintainAspectRatio: false, animation: {duration: 300},
          interaction: {mode: "nearest", intersect: false, axis: "x"},
          plugins: {
            legend: {display: false},
            tooltip: {
              backgroundColor: c.tooltipBg, borderColor: c.border, borderWidth: 1,
              titleColor: c.text, bodyColor: c.muted,
              callbacks: {
                title: items => { const idx = Math.round(items[0].raw.x); return data.dates[idx] ?? ""; },
                label: item => {
                  const idx = Math.round(item.raw.x);
                  return `${item.dataset.label}: ${item.raw.y.toFixed(2)}`;
                }
              }
            }
          },
          scales: {
            x: {
              type: "linear",
              title: {display: true, text: "Game #", color: c.muted, font: {size: 12}},
              ticks: {color: c.muted, font: {size: 11}, maxTicksLimit: 10, callback: v => Math.round(v)},
              grid: {color: c.grid}, border: {color: c.border},
            },
            y: {
              title: {display: true, text: "Points per 100 above avg", color: c.muted, font: {size: 12}},
              ticks: {color: c.muted, font: {size: 11}},
              grid: {color: c.grid}, border: {color: c.border},
            }
          }
        }
      });
    }

    function highlightLeaderboard(name) {
      document.querySelectorAll("#lbBody tr").forEach(tr => {
        const isHL = tr.dataset.player === name;
        tr.classList.toggle("highlighted", isHL);
        if (isHL) tr.scrollIntoView({block: "nearest"});
      });
    }

    function applyLbFilter() {
      const q = document.getElementById("lbSearch").value.trim().toLowerCase();
      lbFiltered = q ? lbData.filter(r => r.player.toLowerCase().includes(q)) : [...lbData];
      sortAndRender();
    }

    function sortAndRender() {
      lbFiltered.sort((a, b) => {
        let va = a[sortCol], vb = b[sortCol];
        if (va == null) return 1; if (vb == null) return -1;
        if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortAsc ? va - vb : vb - va;
      });
      const body = document.getElementById("lbBody");
      body.innerHTML = lbFiltered.map(r => {
        const isHL = r.player === currentPlayer;
        const f = v => v != null ? (v > 0 ? "+" : "") + v.toFixed(1) : "—";
        return `<tr class="${isHL ? "highlighted" : ""}" data-player="${r.player}">
          <td title="${r.player}">${r.player}</td>
          <td class="col-ospm">${f(r.ospm)}</td>
          <td class="col-dspm">${f(r.dspm)}</td>
          <td class="col-spm">${f(r.spm)}</td>
          <td class="col-rapm">${f(r.orapm)}</td>
          <td class="col-rapm">${f(r.drapm)}</td>
          <td class="col-rapm">${f(r.rapm)}</td>
        </tr>`;
      }).join("");
      body.querySelectorAll("tr[data-player]").forEach(tr => {
        tr.addEventListener("click", () => selectPlayer(tr.dataset.player));
      });
      const hl = body.querySelector("tr.highlighted");
      if (hl) hl.scrollIntoView({block: "nearest"});
    }

    document.querySelectorAll("thead th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (sortCol === col) sortAsc = !sortAsc;
        else { sortCol = col; sortAsc = col === "player"; }
        document.querySelectorAll("thead th").forEach(h => {
          h.classList.remove("sorted");
          h.querySelector(".sort-icon").textContent = "↕";
        });
        th.classList.add("sorted");
        th.querySelector(".sort-icon").textContent = sortAsc ? "↑" : "↓";
        sortAndRender();
      });
    });

    document.getElementById("lbSearch").addEventListener("input", applyLbFilter);

    (function() {
      const chartCard = document.querySelector('.chart-card');
      const lbCard    = document.querySelector('.lb-card');
      const lbWrap    = document.querySelector('.lb-table-wrap');
      function sync() {
        const overhead = lbCard.offsetHeight - lbWrap.offsetHeight;
        lbWrap.style.maxHeight = Math.max(200, chartCard.offsetHeight - overhead) + 'px';
      }
      new ResizeObserver(sync).observe(chartCard);
    })();

    sortAndRender();
  </script>
</body>
</html>
"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        RANKINGS_HTML,
        skills=list(SKILLS.keys()),
        skill_abbr=SKILL_ABBR,
    )

@app.route("/explorer")
def explorer():
    all_skill_labels = {**SKILLS, **SPM_SKILLS}
    return render_template_string(
        EXPLORER_HTML,
        players=PLAYERS,
        skills=SKILLS,
        spm_skills=SPM_SKILLS,
        skill_labels=all_skill_labels,
    )

@app.route("/spm")
def spm():
    career_rows = []
    for _, r in spm_career.iterrows():
        career_rows.append({
            "player": r["player"],
            "ospm":   round(float(r["ospm"]),  2),
            "dspm":   round(float(r["dspm"]),  2),
            "spm":    round(float(r["spm"]),   2),
            "orapm":  round(float(r["orapm"]), 2),
            "drapm":  round(float(r["drapm"]), 2),
            "rapm":   round(float(r["rapm"]),  2),
        })
    return render_template_string(
        SPM_HTML,
        players=SPM_PLAYERS,
        career=career_rows,
    )

@app.route("/api/data")
def get_data():
    player = request.args.get("player", "")
    skill  = request.args.get("skill", "pts_per100")
    if skill not in SKILLS and skill not in SPM_SKILLS:
        return jsonify({"error": "unknown skill"}), 400

    if skill in SPM_SKILLS:
        pdata = spm_ts[spm_ts["player"] == player].copy()
    else:
        pdata = df[df["player"] == player].copy()
    if pdata.empty:
        return jsonify({"error": "player not found"}), 404

    vals  = pdata[skill].values
    dates = pdata["game_date"].dt.strftime("%Y-%m-%d").tolist()

    def to_list(arr):
        return [None if np.isnan(v) else round(float(v), 4) for v in arr]

    estimate = to_list(vals)

    xs = np.arange(len(vals), dtype=float)
    mask = ~np.isnan(vals)
    loess_out = [None] * len(vals)
    if mask.sum() >= 4:
        frac = max(0.08, min(0.4, 15.0 / mask.sum()))
        smoothed = lowess(vals[mask], xs[mask], frac=frac, return_sorted=False)
        for i, idx in enumerate(np.where(mask)[0]):
            loess_out[idx] = round(float(smoothed[i]), 4)

    return jsonify({
        "dates":       dates,
        "estimate":    estimate,
        "loess":       loess_out,
        "n_games":     len(pdata),
        "date_range":  f"{dates[0]} – {dates[-1]}" if dates else "",
        "league_avg":  LEAGUE_AVG.get(skill),
    })

@app.route("/api/spm_data")
def get_spm_data():
    player = request.args.get("player", "")
    pdata  = spm_ts[spm_ts["player"] == player].copy()
    if pdata.empty:
        return jsonify({"error": "player not found"}), 404

    dates = pdata["game_date"].dt.strftime("%Y-%m-%d").tolist()

    def to_list(arr):
        return [None if np.isnan(v) else round(float(v), 4) for v in arr]

    def smooth(vals):
        xs   = np.arange(len(vals), dtype=float)
        mask = ~np.isnan(vals)
        out  = [None] * len(vals)
        if mask.sum() >= 4:
            frac = max(0.08, min(0.4, 15.0 / mask.sum()))
            sm   = lowess(vals[mask], xs[mask], frac=frac, return_sorted=False)
            for i, idx in enumerate(np.where(mask)[0]):
                out[idx] = round(float(sm[i]), 4)
        return out

    ospm = pdata["ospm"].values
    dspm = pdata["dspm"].values
    spm  = pdata["spm"].values

    return jsonify({
        "dates":    dates,
        "ospm":     to_list(ospm),
        "dspm":     to_list(dspm),
        "spm":      to_list(spm),
        "ospm_sm":  smooth(ospm),
        "dspm_sm":  smooth(dspm),
        "spm_sm":   smooth(spm),
        "n_games":  len(pdata),
        "date_range": f"{dates[0]} – {dates[-1]}" if dates else "",
    })

@app.route("/api/leaderboard")
def get_leaderboard():
    skill = request.args.get("skill", "pts_per100")
    if skill not in SKILLS and skill not in SPM_SKILLS:
        return jsonify({"error": "unknown skill"}), 400

    if skill in SPM_SKILLS:
        rows = []
        for _, r in spm_career.iterrows():
            val = r.get(skill)
            rows.append({
                "player":  r["player"],
                "value":   round(float(val), 2) if pd.notna(val) else None,
                "n_games": spm_n_games.get(r["player"], 0),
            })
        rows.sort(key=lambda x: (x["value"] is None, -(x["value"] or 0)))
        return jsonify(rows)

    rows = []
    for _, r in LEADERBOARD_DF.iterrows():
        val = r[skill]
        rows.append({
            "player":  r["player"],
            "value":   None if (val is None or (isinstance(val, float) and np.isnan(val))) else round(float(val), 2),
            "n_games": int(r["n_games"]),
        })

    rows.sort(key=lambda x: (x["value"] is None, -(x["value"] or 0)))
    return jsonify(rows)

@app.route("/api/rankings")
def get_rankings():
    spm_df = spm_career[["player", "ospm", "dspm", "spm"]].copy()
    for col in ["ospm", "dspm", "spm"]:
        spm_df[f"{col}_pct"] = (spm_df[col].rank(pct=True, na_option="keep") * 100).round(1)
    spm_lookup = spm_df.set_index("player").to_dict("index")

    rows = []
    for _, r in LEADERBOARD_DF.iterrows():
        row = {"player": r["player"], "n_games": int(r["n_games"])}
        for skill in SKILLS:
            val = r[skill]
            pct = r[f"{skill}_pct"]
            row[skill]          = None if (val is None or (isinstance(val, float) and np.isnan(val))) else round(float(val), 2)
            row[f"{skill}_pct"] = None if (pct is None or (isinstance(pct, float) and np.isnan(pct))) else round(float(pct), 1)
        spm_row = spm_lookup.get(r["player"], {})
        for col in ["ospm", "dspm", "spm"]:
            v = spm_row.get(col)
            p = spm_row.get(f"{col}_pct")
            row[col]          = round(float(v), 2) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            row[f"{col}_pct"] = round(float(p), 1) if p is not None and not (isinstance(p, float) and np.isnan(p)) else None
        rows.append(row)

    rows.sort(key=lambda x: (x["pts_per100"] is None, -(x["pts_per100"] or 0)))
    return jsonify(rows)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
