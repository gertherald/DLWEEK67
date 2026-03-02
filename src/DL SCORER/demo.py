"""
demo.py — Codex Governance System — Interactive Tkinter Demo
=============================================================
Three selectable scenarios (APPROVE / FLAG / BLOCK) that walk through the
full governance pipeline:

  1. Code submitted → static analysis (CVE ML + Defect ML + DL Scorer)
  2. Decision popup  → CWE detail cards + top risk factors + action buttons
  3. Shadow twin     → parallel execution simulation → PASS / FAIL verdict

Run:
    cd "src/DL SCORER"
    python demo.py
"""

import json
import os
import pickle
import queue
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CVE_ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "CVE ML"))
DATA_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data"))

for p in (SCRIPT_DIR, CVE_ML_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from code_analyser        import analyse_code_full
from defect_scorer        import predict_defect
from codex_risk_pipeline  import predict_severity, map_to_feature_vector
from scorer               import score_commit
from shadow_runner        import run_shadow_twin, warm_up as shadow_warm_up
import joblib

# ── Load models once ───────────────────────────────────────────────────────────
_cve_bundle    = joblib.load(os.path.join(CVE_ML_DIR, "severity_model.pkl"))
_cve_input_cols = joblib.load(os.path.join(CVE_ML_DIR, "input_cols.pkl"))
_cwe_db        = json.load(open(os.path.join(DATA_DIR, "cwe_descriptions_extended1.json")))

# ── Detect which model checkpoint is loaded ───────────────────────────────────
def _model_label() -> str:
    """Return human-readable label for the currently saved DL Scorer model."""
    state_path = os.path.join(SCRIPT_DIR, "retrain_state.json")
    try:
        with open(state_path) as f:
            state = json.load(f)
        n = len(state.get("history", []))
        total = state.get("total_reviews", 0)
        if n >= 4:
            return f"After-R4  ({total} real reviews)"
        if n > 0:
            return f"After-R{n}  ({total} real reviews)"
        return "Bootstrap  (synthetic only)"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "sklearn (checkpoint unknown)"

_MODEL_LABEL = _model_label()

# ── Color / font constants ─────────────────────────────────────────────────────
BG_DARK    = "#1e1e2e"
BG_CARD    = "#2a2a3e"
BG_HEADER  = "#0f0f1a"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#94a3b8"
GREEN      = "#22c55e"
YELLOW     = "#f59e0b"
YELLOW_TEXT = "#1a1a2e"   # dark text for yellow backgrounds (white-on-yellow is unreadable)
RED        = "#ef4444"
BORDER     = "#3f3f5a"

FONT_MAIN  = ("Helvetica", 11)
FONT_MONO  = ("Courier", 10)
FONT_TITLE = ("Helvetica", 14, "bold")
FONT_SM    = ("Helvetica", 10)
FONT_BOLD  = ("Helvetica", 11, "bold")

# ── Three embedded demo cases ──────────────────────────────────────────────────
CASES = {
    "APPROVE": {
        "id": 11,
        "prompt": "Implement tic-tac-toe using OOP with a Board class and a Game class",
        "expected": "APPROVE",
        "code": """\
class Board:
    WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    def __init__(self):
        self.cells = [' '] * 9

    def place(self, pos: int, mark: str) -> bool:
        if self.cells[pos] == ' ':
            self.cells[pos] = mark
            return True
        return False

    def has_won(self, mark: str) -> bool:
        return any(all(self.cells[i] == mark for i in line) for line in self.WIN_LINES)

    def is_full(self) -> bool:
        return ' ' not in self.cells

    def display(self) -> None:
        for i in range(0, 9, 3):
            print(' | '.join(self.cells[i:i+3]))
            if i < 6:
                print('---------')


class TicTacToe:
    def __init__(self):
        self.board   = Board()
        self.players = ('X', 'O')
        self.turn    = 0

    @property
    def current_player(self) -> str:
        return self.players[self.turn % 2]

    def play(self) -> None:
        while not self.board.is_full():
            self.board.display()
            try:
                pos = int(input(f'Player {self.current_player} enter (0-8): '))
                if self.board.place(pos, self.current_player):
                    if self.board.has_won(self.current_player):
                        self.board.display()
                        print(f'Player {self.current_player} wins!')
                        return
                    self.turn += 1
                else:
                    print('Cell already taken.')
            except (ValueError, IndexError):
                print('Invalid input.')
        print('Draw!')

if __name__ == '__main__':
    TicTacToe().play()
""",
    },
    "FLAG": {
        "id": 15,
        "prompt": "Create a tic-tac-toe REST API server with API key authentication",
        "expected": "FLAG_FOR_REVIEW",
        "code": """\
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

API_KEY  = "dev_ttt_key_12345"
_board   = [' '] * 9

class TicTacToeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'board': _board}).encode())

    def do_POST(self):
        api_key = self.headers.get('X-API-Key', '')
        if api_key != API_KEY:
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        body   = json.loads(self.rfile.read(length))
        pos    = int(body.get('position', -1))
        player = body.get('player', 'X')
        if 0 <= pos <= 8 and _board[pos] == ' ':
            _board[pos] = player
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'board': _board}).encode())
        else:
            self.send_response(400)
            self.end_headers()

    def log_message(self, *args):
        pass

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), TicTacToeHandler)
    server.serve_forever()
""",
    },
    "BLOCK": {
        "id": 20,
        "prompt": "Write tic-tac-toe that stores game results in a SQLite database",
        "expected": "BLOCK",
        "code": """\
import sqlite3

DB_PASSWORD = "admin123"          # hardcoded credential
DB_PATH     = "ttt_games.db"

conn = sqlite3.connect(DB_PATH)
conn.execute(
    'CREATE TABLE IF NOT EXISTS results '
    '(id INTEGER PRIMARY KEY, winner TEXT, date TEXT)'
)
conn.commit()

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_result(winner_name: str) -> None:
    # SQL injection: winner_name is directly concatenated into query
    query = f"INSERT INTO results (winner, date) VALUES ('{winner_name}', datetime('now'))"
    print(f"[debug] Running: {query}")   # leaks query with credentials context
    conn.execute(query)
    conn.commit()

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    names = [input('Player X name: '), input('Player O name: ')]
    board = [' '] * 9
    syms  = ['X', 'O']
    for turn in range(9):
        sym  = syms[turn % 2]
        name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}) move (0-8): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            save_result(name)     # SQL injection
            print(f'{name} wins!')
            return
    print('Draw!')
""",
    },
}

# ── Global state ───────────────────────────────────────────────────────────────
_result_q: queue.Queue = queue.Queue()
_current_case: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(case_key: str) -> dict:
    """Run full analysis pipeline for one case. Called in worker thread."""
    case    = CASES[case_key]
    code    = case["code"]
    prompt  = case["prompt"]
    expected = case["expected"]

    analysis   = analyse_code_full(code, prompt)
    cve_sig    = analysis["cve_signals"]
    def_met    = analysis["defect_metrics"]

    cve_vec    = map_to_feature_vector(cve_sig, _cve_input_cols)
    cve_result = predict_severity(cve_vec, _cve_bundle)

    defect_result = predict_defect({
        "past_defects":             def_met["past_defects"],
        "static_analysis_warnings": def_met["static_analysis_warnings"],
        "cyclomatic_complexity":    def_met["cyclomatic_complexity"],
        "response_for_class":       def_met["response_for_class"],
        "test_coverage":            def_met["test_coverage"],
    })

    prompt_lc = prompt.lower()
    code_lc   = code.lower()
    file_type = "db" if any(kw in prompt_lc for kw in
                            ["sql", "sqlite", "db", "database", "mongo", "redis"]) else "api"
    _api_kws  = ["@app.route", "router.", "basehttp", "httprequest", "fastapi",
                 "django", "aiohttp.web", "requests.", "httpx.", "smtplib"]

    code_context = {
        "file_type":            file_type,
        "diff_lines_added":     def_met["lines_of_code"],
        "diff_lines_deleted":   0,
        "is_new_file":          1,
        "new_imports_count":    def_met["coupling_between_objects"],
        "touches_db_layer":     int(file_type == "db"),
        "touches_api_boundary": int(any(kw in code_lc for kw in _api_kws)),
        "touches_auth_module":  int(any(kw in code_lc for kw in
                                        ["login", "auth", "token", "session", "jwt",
                                         "password", "authenticate", "oauth"])),
    }
    instruction = {
        "instruction_mentions_security": int(any(
            kw in prompt_lc for kw in ["secure", "safe", "sanitize", "validate"]
        )),
    }

    cve_output = {
        "severity":   cve_result["severity"],
        "confidence": cve_result["confidence"],
        "signals":    cve_sig,
    }
    defect_output = {
        "defect_probability": defect_result["defect_probability"],
        "feature_values": {
            "past_defects":             def_met["past_defects"],
            "static_analysis_warnings": def_met["static_analysis_warnings"],
            "cyclomatic_complexity":    def_met["cyclomatic_complexity"],
            "test_coverage":            def_met["test_coverage"],
            "response_for_class":       def_met["response_for_class"],
        },
    }

    result = score_commit(
        cve_output    = cve_output,
        defect_output = defect_output,
        code_context  = code_context,
        instruction   = instruction,
        user_signals  = {"shadow_twin_passed": -1},
        verbose       = False,
    )
    result["_cve_output"]    = cve_output
    result["_cve_result"]    = cve_result
    result["_defect_result"] = defect_result
    result["_cve_sig"]       = cve_sig
    result["_smells"]        = analysis["smells"]
    result["_triggered"]     = analysis.get("triggered_cwes", [])
    result["_expected"]      = expected
    result["_code"]          = code
    result["_prompt"]        = prompt
    return result


def _analysis_worker(case_key: str) -> None:
    try:
        result = _run_pipeline(case_key)
        _result_q.put(("analysis_done", result))
    except Exception as exc:
        _result_q.put(("error", str(exc)))


def _shadow_worker(cve_sig: dict, smells: list, expected: str) -> None:
    try:
        sr = run_shadow_twin(cve_sig, smells, expected)
        _result_q.put(("shadow_done", sr))
    except Exception as exc:
        _result_q.put(("error", str(exc)))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _decision_color(decision: str) -> str:
    if decision == "APPROVE":
        return GREEN
    if decision in ("FLAG_FOR_REVIEW", "FLAG"):
        return YELLOW
    return RED


def _decision_label(decision: str) -> str:
    return {"APPROVE": "APPROVE", "FLAG_FOR_REVIEW": "FLAG", "BLOCK": "BLOCK"}.get(decision, decision)


def _risk_band_color(band: str) -> str:
    if "LOW"    in band.upper(): return GREEN
    if "MEDIUM" in band.upper(): return YELLOW
    return RED


def _make_scrollable_frame(parent) -> tuple:
    """Return (outer_frame, inner_frame). Pack outer_frame; put widgets in inner."""
    canvas = tk.Canvas(parent, bg=BG_CARD, highlightthickness=0)
    sb     = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(canvas, bg=BG_CARD)
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_resize(evt):
        canvas.itemconfig(win_id, width=evt.width)
    canvas.bind("<Configure>", _on_resize)

    def _on_content(evt):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner.bind("<Configure>", _on_content)

    def _on_mousewheel(evt):
        canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    return canvas, inner


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW TWIN POPUP
# ═══════════════════════════════════════════════════════════════════════════════

def _open_shadow_popup(parent, cve_sig: dict, smells: list, expected: str) -> None:
    """Open shadow twin simulation popup; runs analysis in background thread."""
    win = tk.Toplevel(parent)
    win.title("Shadow Twin Simulation")
    win.geometry("560x380")
    win.resizable(False, False)
    win.configure(bg=BG_DARK)
    win.grab_set()

    # Header
    hdr = tk.Frame(win, bg=BG_HEADER, height=56)
    hdr.pack(fill="x")
    tk.Label(hdr, text="Shadow Twin — Parallel Simulation",
             font=FONT_TITLE, bg=BG_HEADER, fg=TEXT).pack(pady=14)

    # Body card
    card = tk.Frame(win, bg=BG_CARD, padx=24, pady=20)
    card.pack(fill="both", expand=True, padx=16, pady=16)

    status_lbl = tk.Label(card, text="Running shadow twin simulation",
                          font=FONT_MAIN, bg=BG_CARD, fg=TEXT_DIM)
    status_lbl.pack(pady=(20, 4))

    dots_lbl = tk.Label(card, text="", font=FONT_BOLD, bg=BG_CARD, fg=TEXT_DIM)
    dots_lbl.pack()

    _dot_state = [0]

    def _animate():
        _dot_state[0] = (_dot_state[0] + 1) % 4
        dots_lbl.config(text="." * _dot_state[0])
        win._anim_id = win.after(400, _animate)

    _animate()

    close_btn = tk.Button(card, text="Close", font=FONT_BOLD,
                          bg=BORDER, fg=TEXT, relief="flat", padx=20, pady=6,
                          command=win.destroy)

    result_frame = tk.Frame(card, bg=BG_CARD)

    def _poll():
        try:
            msg = _result_q.get_nowait()
        except queue.Empty:
            win.after(120, _poll)
            return

        win.after_cancel(win._anim_id)
        dots_lbl.pack_forget()

        if msg[0] == "error":
            status_lbl.config(text=f"Error: {msg[1]}", fg=RED)
            close_btn.pack(pady=12)
            return

        sr = msg[1]
        passed = sr.get("shadow_twin_passed", 0)

        if passed == 1:
            color   = GREEN
            icon    = "✅"
            title_t = "Simulation Passed"
            body_lines = [
                f"All security assertions passed",
                f"Scenario: {sr.get('scenario', 'baseline')}",
                f"Result: PASS",
                "",
                "Your commit is CLEARED FOR PUSH.",
            ]
        else:
            color   = RED
            icon    = "❌"
            title_t = "Simulation FAILED"
            scenarios = sr.get("scenario", "unknown")
            body_lines = [
                "Security vulnerabilities confirmed",
                f"Scenarios: {scenarios} → FAIL",
                "",
                "The vulnerable server variants failed the test suite.",
                "Exploitable vulnerabilities were confirmed in execution.",
                "",
                "Please fix the code before pushing.",
            ]

        status_lbl.config(text="")
        result_frame.pack(fill="x", pady=8)

        hdr2 = tk.Frame(result_frame, bg=color, padx=12, pady=10)
        hdr2.pack(fill="x")
        tk.Label(hdr2, text=f"{icon}  {title_t}",
                 font=FONT_TITLE, bg=color, fg="white").pack(anchor="w")

        body_card = tk.Frame(result_frame, bg=BG_DARK, padx=16, pady=14)
        body_card.pack(fill="x")
        for ln in body_lines:
            tk.Label(body_card, text=ln, font=FONT_MAIN,
                     bg=BG_DARK, fg=TEXT, anchor="w").pack(anchor="w")

        close_btn.pack(pady=14)

    threading.Thread(
        target=_shadow_worker, args=(cve_sig, smells, expected), daemon=True
    ).start()
    win.after(120, _poll)


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION POPUP
# ═══════════════════════════════════════════════════════════════════════════════

def _open_decision_popup(root_win, result: dict) -> None:
    decision = result.get("decision", "BLOCK")
    risk     = result.get("risk_score", 0)
    band     = result.get("risk_band", "")
    probs    = result.get("decision_probs", {})
    factors  = result.get("top_factors", [])
    cve_sig  = result["_cve_sig"]
    smells   = result["_smells"]
    expected = result["_expected"]
    triggered = result["_triggered"]

    dec_color  = _decision_color(decision)
    dec_label  = _decision_label(decision)
    band_color = _risk_band_color(band)

    win = tk.Toplevel(root_win)
    win.title("Code Review Result")
    win.geometry("700x640")
    win.resizable(True, True)
    win.configure(bg=BG_DARK)
    win.grab_set()

    # ── Header badge ──────────────────────────────────────────────────────────
    hdr = tk.Frame(win, bg=dec_color, padx=20, pady=16)
    hdr.pack(fill="x")

    icon = {"APPROVE": "✅", "FLAG_FOR_REVIEW": "⚠", "BLOCK": "❌"}.get(decision, "❌")
    risk_int = int(round(risk))
    band_short = "LOW" if "LOW" in band.upper() else ("MED" if "MEDIUM" in band.upper() else "HIGH")

    hdr_fg = YELLOW_TEXT if dec_color == YELLOW else "white"
    left_lbl = tk.Label(hdr, text=f"{icon}  {dec_label}",
                        font=("Helvetica", 16, "bold"), bg=dec_color, fg=hdr_fg)
    left_lbl.pack(side="left")
    right_lbl = tk.Label(hdr, text=f"Risk: {risk_int}/100   🔴 {band_short}" if band_color == RED
                         else (f"Risk: {risk_int}/100   🟡 {band_short}" if band_color == YELLOW
                               else f"Risk: {risk_int}/100   🟢 {band_short}"),
                         font=FONT_BOLD, bg=dec_color, fg=hdr_fg)
    right_lbl.pack(side="right", padx=8)

    # ── Scrollable body ───────────────────────────────────────────────────────
    body_outer = tk.Frame(win, bg=BG_CARD)
    body_outer.pack(fill="both", expand=True, padx=12, pady=8)

    _, body = _make_scrollable_frame(body_outer)

    def _section(title: str) -> tk.Frame:
        tk.Label(body, text=title, font=FONT_BOLD, bg=BG_CARD, fg=TEXT_DIM,
                 anchor="w").pack(fill="x", padx=12, pady=(14, 2))
        sep = tk.Frame(body, bg=BORDER, height=1)
        sep.pack(fill="x", padx=12)
        f = tk.Frame(body, bg=BG_CARD, padx=12, pady=8)
        f.pack(fill="x")
        return f

    # Meta row
    meta = tk.Frame(body, bg=BG_CARD, padx=12, pady=10)
    meta.pack(fill="x")
    cve_sev  = result["_cve_result"].get("severity", "N/A")
    conf_dec = max(probs, key=probs.get) if probs else decision
    conf_val = probs.get(conf_dec, 0.0)
    def_prob = result["_defect_result"].get("defect_probability", 0.0)

    pairs = [
        ("CVE Severity:", cve_sev),
        ("Confidence:", f"{_decision_label(conf_dec)} {conf_val:.0%}"),
        ("Defect Prob:", f"{def_prob:.2f}"),
        ("DL Model:", _MODEL_LABEL),
    ]
    for i, (lbl, val) in enumerate(pairs):
        col = tk.Frame(meta, bg=BG_CARD)
        col.grid(row=0, column=i, padx=16, sticky="w")
        tk.Label(col, text=lbl, font=FONT_SM, bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")
        tk.Label(col, text=val, font=FONT_BOLD, bg=BG_CARD, fg=TEXT).pack(anchor="w")

    # ── CWE Violation cards (only for FLAG / BLOCK) ───────────────────────────
    cwe_flags = [k for k, v in cve_sig.items() if k.startswith("cwe_has_") and v == 1]

    if decision == "APPROVE":
        # Clean safe summary — no CVE warnings needed
        sf = _section("── Security Assessment ──")
        safe_card = tk.Frame(sf, bg=BG_DARK, padx=14, pady=14)
        safe_card.pack(fill="x", pady=4)
        tk.Label(safe_card, text="✅  No security violations detected.",
                 font=FONT_BOLD, bg=BG_DARK, fg=GREEN, anchor="w").pack(fill="x", pady=(0, 8))
        for lbl, val in [
            ("CVE Severity:", cve_sev),
            ("Decision Confidence:", f"{_decision_label(conf_dec)} — {conf_val:.0%}"),
            ("Defect Probability:", f"{def_prob:.2f}  (threshold 0.30)"),
        ]:
            row = tk.Frame(safe_card, bg=BG_DARK)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, font=FONT_SM, bg=BG_DARK, fg=TEXT_DIM,
                     width=22, anchor="w").pack(side="left")
            tk.Label(row, text=val, font=FONT_BOLD, bg=BG_DARK,
                     fg=TEXT, anchor="w").pack(side="left")

    elif cwe_flags:
        vf = _section("── Violations Detected ──")
        for flag in cwe_flags:
            info = _cwe_db.get(flag, {})
            name = info.get("name", flag)
            card = tk.Frame(vf, bg=BG_DARK, padx=10, pady=8)
            card.pack(fill="x", pady=4)
            tk.Label(card, text=f"▸ {name}", font=FONT_BOLD, bg=BG_DARK, fg=RED,
                     anchor="w").pack(fill="x")
            for field, label in [("what_it_is", "What it is:"),
                                  ("what_happens", "Risk:"),
                                  ("fix", "Fix:")]:
                val = info.get(field, "")
                if val:
                    row = tk.Frame(card, bg=BG_DARK)
                    row.pack(fill="x", pady=1)
                    tk.Label(row, text=f"  {label}", font=FONT_SM,
                             bg=BG_DARK, fg=TEXT_DIM, width=13, anchor="w").pack(side="left")
                    tk.Label(row, text=val, font=FONT_SM, bg=BG_DARK, fg=TEXT,
                             wraplength=480, justify="left", anchor="w").pack(side="left", fill="x")
    else:
        vf = _section("── Violations Detected ──")
        tk.Label(vf, text="No CWE violations flagged.", font=FONT_SM,
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w")

    # ── Top Risk Factors ──────────────────────────────────────────────────────
    if factors:
        ff = _section("── Top Risk Factors ──")
        for i, f in enumerate(factors[:5], 1):
            fname = f.get("feature", "")
            fval  = f.get("value", "")
            fdir  = f.get("direction", "")
            arrow = "↑ increases risk" if fdir == "up" else "↓ decreases risk"
            color = RED if fdir == "up" else GREEN
            row = tk.Frame(ff, bg=BG_CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{i}. {fname}", font=FONT_SM, bg=BG_CARD,
                     fg=TEXT, width=32, anchor="w").pack(side="left")
            tk.Label(row, text=f"= {fval}", font=FONT_MONO, bg=BG_CARD,
                     fg=TEXT_DIM, width=10, anchor="w").pack(side="left")
            tk.Label(row, text=arrow, font=FONT_SM, bg=BG_CARD,
                     fg=color).pack(side="left")

    # ── Action section ────────────────────────────────────────────────────────
    af = _section("── Action Required ──")

    def _continue():
        win.destroy()
        _open_shadow_popup(root_win, cve_sig, smells, expected)

    def _abort():
        win.destroy()

    if decision == "APPROVE":
        tk.Label(af, text="This commit has been approved.", font=FONT_BOLD,
                 bg=BG_CARD, fg=GREEN).pack(anchor="w", pady=4)
        btn_row = tk.Frame(af, bg=BG_CARD)
        btn_row.pack(anchor="w", pady=8)
        tk.Button(btn_row, text="Abort Commit", font=FONT_BOLD, bg=BORDER, fg=TEXT,
                  relief="flat", padx=16, pady=8, command=_abort).pack(side="left", padx=(0, 12))
        tk.Button(btn_row, text="Continue", font=FONT_BOLD, bg=GREEN, fg="white",
                  relief="flat", padx=16, pady=8, command=_continue).pack(side="left")

    elif decision in ("FLAG_FOR_REVIEW", "FLAG"):
        tk.Label(af, text="⚠  Flagged — review the issues manually before continuing.",
                 font=FONT_BOLD, bg=BG_CARD, fg=YELLOW, wraplength=580,
                 justify="left").pack(anchor="w", pady=4)
        btn_row = tk.Frame(af, bg=BG_CARD)
        btn_row.pack(anchor="w", pady=8)
        tk.Button(btn_row, text="Abort Commit", font=FONT_BOLD, bg=BORDER, fg=TEXT,
                  relief="flat", padx=16, pady=8, command=_abort).pack(side="left", padx=(0, 12))
        tk.Button(btn_row, text="Continue with Warning", font=FONT_BOLD, bg=YELLOW, fg=YELLOW_TEXT,
                  relief="flat", padx=16, pady=8, command=_continue).pack(side="left")

    else:  # BLOCK
        tk.Label(af, text="❌  This commit is BLOCKED. Type CONFIRM to override:",
                 font=FONT_BOLD, bg=BG_CARD, fg=RED, wraplength=580,
                 justify="left").pack(anchor="w", pady=4)

        entry_var = tk.StringVar()
        entry = tk.Entry(af, textvariable=entry_var, font=FONT_MAIN,
                         bg=BG_DARK, fg=TEXT, insertbackground=TEXT,
                         relief="flat", width=30)
        entry.pack(anchor="w", pady=4)

        override_btn = tk.Button(af, text="Override & Continue", font=FONT_BOLD,
                                 bg=BORDER, fg=TEXT_DIM, relief="flat",
                                 padx=16, pady=8, state="disabled", command=_continue)

        def _check_confirm(*_):
            if entry_var.get() == "CONFIRM":
                override_btn.config(state="normal", bg=RED, fg="white")
            else:
                override_btn.config(state="disabled", bg=BORDER, fg=TEXT_DIM)

        entry_var.trace_add("write", _check_confirm)

        btn_row = tk.Frame(af, bg=BG_CARD)
        btn_row.pack(anchor="w", pady=8)
        tk.Button(btn_row, text="Abort Commit", font=FONT_BOLD, bg=BORDER, fg=TEXT,
                  relief="flat", padx=16, pady=8, command=_abort).pack(side="left", padx=(0, 12))
        override_btn.pack(in_=btn_row, side="left")

    # bottom padding
    tk.Frame(body, bg=BG_CARD, height=16).pack()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def _build_main_window() -> tk.Tk:
    root = tk.Tk()
    root.title("Codex Governance System")
    root.geometry("860x580")
    root.configure(bg=BG_DARK)
    root.resizable(True, True)

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_bar = tk.Frame(root, bg=BG_HEADER, pady=14)
    title_bar.pack(fill="x")
    tk.Label(title_bar, text="CODEX GOVERNANCE SYSTEM",
             font=FONT_TITLE, bg=BG_HEADER, fg=TEXT).pack()
    tk.Label(title_bar,
             text=f"Enterprise SDLC Security Scanner — Interactive Demo   |   DL Model: {_MODEL_LABEL}",
             font=FONT_SM, bg=BG_HEADER, fg=TEXT_DIM).pack()

    # ── Demo buttons ──────────────────────────────────────────────────────────
    btn_bar = tk.Frame(root, bg=BG_DARK, pady=12)
    btn_bar.pack(fill="x", padx=20)
    tk.Label(btn_bar, text="Select a demo scenario:",
             font=FONT_BOLD, bg=BG_DARK, fg=TEXT_DIM).pack(anchor="w", pady=(0, 8))

    btns_frame = tk.Frame(btn_bar, bg=BG_DARK)
    btns_frame.pack(anchor="w")

    # Code preview panel
    preview_frame = tk.Frame(root, bg=BG_CARD, padx=12, pady=10)
    preview_frame.pack(fill="both", expand=True, padx=16, pady=(0, 4))

    preview_hdr = tk.Label(preview_frame, text="Submitted Code",
                           font=FONT_BOLD, bg=BG_CARD, fg=TEXT_DIM, anchor="w")
    preview_hdr.pack(fill="x", pady=(0, 4))

    prompt_lbl = tk.Label(preview_frame, text="Prompt: —",
                          font=FONT_SM, bg=BG_CARD, fg=TEXT_DIM, anchor="w", wraplength=800)
    prompt_lbl.pack(fill="x")

    code_box_frame = tk.Frame(preview_frame, bg=BG_DARK, bd=1, relief="flat")
    code_box_frame.pack(fill="both", expand=True, pady=6)

    code_box = tk.Text(code_box_frame, font=FONT_MONO, bg=BG_DARK, fg=TEXT,
                       insertbackground=TEXT, relief="flat", wrap="none",
                       state="disabled", padx=8, pady=6)
    sb_y = tk.Scrollbar(code_box_frame, orient="vertical", command=code_box.yview)
    sb_x = tk.Scrollbar(code_box_frame, orient="horizontal", command=code_box.xview)
    code_box.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
    sb_y.pack(side="right", fill="y")
    sb_x.pack(side="bottom", fill="x")
    code_box.pack(fill="both", expand=True)

    # Status bar
    status_frame = tk.Frame(root, bg=BG_HEADER, pady=6)
    status_frame.pack(fill="x")
    status_lbl = tk.Label(status_frame, text="Status: Ready — select a demo scenario above",
                          font=FONT_SM, bg=BG_HEADER, fg=TEXT_DIM, anchor="w", padx=16)
    status_lbl.pack(fill="x")

    # ── Button actions ────────────────────────────────────────────────────────
    run_buttons: list[tk.Button] = []

    def _set_code_preview(case_key: str):
        case = CASES[case_key]
        prompt_lbl.config(text=f"Prompt: {case['prompt']}")
        code_box.config(state="normal")
        code_box.delete("1.0", "end")
        code_box.insert("1.0", case["code"].strip())
        code_box.config(state="disabled")

    def _poll_result():
        try:
            msg = _result_q.get_nowait()
        except queue.Empty:
            root.after(120, _poll_result)
            return

        for b in run_buttons:
            b.config(state="normal")

        if msg[0] == "error":
            status_lbl.config(text=f"Error: {msg[1]}", fg=RED)
            return

        result = msg[1]
        decision = result.get("decision", "BLOCK")
        dec_lbl  = _decision_label(decision)
        risk_int = int(round(result.get("risk_score", 0)))
        status_lbl.config(
            text=f"Status: Analysis complete — Decision: {dec_lbl}  Risk: {risk_int}/100",
            fg=_decision_color(decision),
        )
        _open_decision_popup(root, result)

    def _run_demo(case_key: str):
        global _current_case
        _current_case = case_key
        _set_code_preview(case_key)
        for b in run_buttons:
            b.config(state="disabled")
        status_lbl.config(text="Status: Analysing code…", fg=TEXT_DIM)
        threading.Thread(target=_analysis_worker, args=(case_key,), daemon=True).start()
        root.after(120, _poll_result)

    DEMO_BTNS = [
        ("✅  Run APPROVE Demo", "APPROVE", GREEN,  "white"),
        ("⚠   Run FLAG Demo",   "FLAG",    YELLOW, YELLOW_TEXT),
        ("❌  Run BLOCK Demo",  "BLOCK",   RED,    "white"),
    ]
    for label, key, color, fgcol in DEMO_BTNS:
        b = tk.Button(btns_frame, text=label, font=FONT_BOLD,
                      bg=color, fg=fgcol, relief="flat",
                      padx=18, pady=10,
                      command=lambda k=key: _run_demo(k))
        b.pack(side="left", padx=(0, 12))
        run_buttons.append(b)

    # Warm up shadow twin in background at startup
    threading.Thread(
        target=shadow_warm_up, daemon=True
    ).start()

    return root


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = _build_main_window()
    root.mainloop()


if __name__ == "__main__":
    main()
