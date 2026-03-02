"""
test_tic_tac_toe.py — End-to-end pipeline test with 10 Tic-Tac-Toe prompts
============================================================================
Tests the full analysis stack:
  1. code_analyser   → CVE signals + defect metrics from raw code
  2. CVE ML          → CVSS severity + confidence
  3. Defect ML       → defect probability %
  4. DL feature vector (feature_builder)
  5. DL Scorer       → risk score + APPROVE / FLAG_FOR_REVIEW / BLOCK
                       (requires PyTorch; gracefully skipped if unavailable)

After running all 10 cases, results are:
  - Printed in a formatted comparison table
  - Saved to ttt_test_results.csv for DL retraining (python train.py --data ...)
  - DL learning is demonstrated by re-running train.py on the augmented dataset

Run:
  cd "src/DL SCORER"
  python test_tic_tac_toe.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CVE_ML_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "CVE ML"))
RESULTS_CSV  = os.path.join(SCRIPT_DIR, "ttt_test_results.csv")

if CVE_ML_DIR not in sys.path:
    sys.path.insert(0, CVE_ML_DIR)

from code_analyser   import analyse_code_full
from defect_scorer   import predict_defect
from feature_builder import build_feature_vector, feature_vector_to_array, FEATURE_NAMES, DECISION_MAP
from codex_risk_pipeline import predict_severity, map_to_feature_vector


# ═══════════════════════════════════════════════════════════════════════════════
# 10 TEST CASES
# Format: (id, prompt, expected_decision, code)
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: list[tuple] = [

    # ── CASE 1: Clean two-player CLI game ─────────────────────────────────────
    (1, "Write a clean tic-tac-toe game in Python for two players", "APPROVE", """\
def display_board(board):
    for i, row in enumerate(board):
        print(" | ".join(row))
        if i < 2:
            print("-" * 9)

def check_winner(board, player):
    lines = [board[r] for r in range(3)]                       # rows
    lines += [[board[r][c] for r in range(3)] for c in range(3)]  # cols
    lines += [[board[i][i] for i in range(3)]]                 # diagonal
    lines += [[board[i][2 - i] for i in range(3)]]             # anti-diagonal
    return any(all(c == player for c in line) for line in lines)

def get_move(board, player):
    while True:
        try:
            move = int(input(f"Player {player}, enter position (1-9): ")) - 1
            row, col = divmod(move, 3)
            if 0 <= move <= 8 and board[row][col] == " ":
                return row, col
            print("That cell is taken or out of range. Try again.")
        except ValueError:
            print("Please enter a number between 1 and 9.")

def play():
    board = [[" "] * 3 for _ in range(3)]
    players = ["X", "O"]
    for turn in range(9):
        player = players[turn % 2]
        display_board(board)
        row, col = get_move(board, player)
        board[row][col] = player
        if check_winner(board, player):
            display_board(board)
            print(f"\\nPlayer {player} wins! Congratulations!")
            return
    display_board(board)
    print("\\nIt's a draw!")

if __name__ == "__main__":
    play()
"""),

    # ── CASE 2: Type-annotated game with comprehensive validation ─────────────
    (2, "Write a tic-tac-toe game in Python with type annotations and input validation",
     "APPROVE", """\
from typing import Optional

Board = list[list[str]]

def create_board() -> Board:
    \"\"\"Return an empty 3x3 board.\"\"\"
    return [[" " for _ in range(3)] for _ in range(3)]

def is_valid_move(board: Board, row: int, col: int) -> bool:
    \"\"\"Return True only if (row, col) is in bounds and unoccupied.\"\"\"
    return 0 <= row < 3 and 0 <= col < 3 and board[row][col] == " "

def apply_move(board: Board, row: int, col: int, player: str) -> Board:
    \"\"\"Return a new board with the move applied (immutable update).\"\"\"
    new_board = [r[:] for r in board]
    new_board[row][col] = player
    return new_board

def get_winner(board: Board) -> Optional[str]:
    \"\"\"Return winning player symbol, or None if no winner yet.\"\"\"
    for player in ("X", "O"):
        for r in range(3):
            if all(board[r][c] == player for c in range(3)):
                return player
        for c in range(3):
            if all(board[r][c] == player for r in range(3)):
                return player
        if all(board[i][i] == player for i in range(3)):
            return player
        if all(board[i][2 - i] == player for i in range(3)):
            return player
    return None

def is_draw(board: Board) -> bool:
    return all(board[r][c] != " " for r in range(3) for c in range(3))

def parse_input(raw: str) -> tuple[int, int]:
    \"\"\"Parse 'row,col' input. Raises ValueError on bad format.\"\"\"
    parts = raw.strip().split(",")
    if len(parts) != 2:
        raise ValueError("Expected format: row,col")
    return int(parts[0]), int(parts[1])

def play_game() -> None:
    board   = create_board()
    players = ("X", "O")
    turn    = 0
    while True:
        player = players[turn % 2]
        print(f"\\nPlayer {player}'s turn")
        print("\\n".join(" | ".join(row) for row in board))
        try:
            row, col = parse_input(input("Enter row,col (0-indexed): "))
        except ValueError as e:
            print(f"Invalid input: {e}. Try again.")
            continue
        if not is_valid_move(board, row, col):
            print("Cell occupied or out of bounds. Try again.")
            continue
        board  = apply_move(board, row, col, player)
        winner = get_winner(board)
        if winner:
            print(f"\\nPlayer {winner} wins!")
            break
        if is_draw(board):
            print("\\nDraw!")
            break
        turn += 1

if __name__ == "__main__":
    play_game()
"""),

    # ── CASE 3: Minimax AI opponent ───────────────────────────────────────────
    (3, "Write a tic-tac-toe game in Python with an unbeatable minimax AI opponent",
     "APPROVE", """\
import math

def minimax(board, is_maximising, depth=0):
    \"\"\"Minimax without alpha-beta for clarity. AI = X, human = O.\"\"\"
    winner = check_winner(board)
    if winner == "X": return  10 - depth
    if winner == "O": return -10 + depth
    if is_board_full(board): return 0

    scores = []
    for r in range(3):
        for c in range(3):
            if board[r][c] == " ":
                board[r][c] = "X" if is_maximising else "O"
                scores.append(minimax(board, not is_maximising, depth + 1))
                board[r][c] = " "
    return max(scores) if is_maximising else min(scores)

def best_move(board):
    \"\"\"Return (row, col) for the best AI move.\"\"\"
    best_score, best_pos = -math.inf, None
    for r in range(3):
        for c in range(3):
            if board[r][c] == " ":
                board[r][c] = "X"
                score = minimax(board, False)
                board[r][c] = " "
                if score > best_score:
                    best_score, best_pos = score, (r, c)
    return best_pos

def check_winner(board):
    for player in ("X", "O"):
        for i in range(3):
            if all(board[i][j] == player for j in range(3)):
                return player
            if all(board[j][i] == player for j in range(3)):
                return player
        if all(board[i][i] == player for i in range(3)):
            return player
        if all(board[i][2-i] == player for i in range(3)):
            return player
    return None

def is_board_full(board):
    return all(board[r][c] != " " for r in range(3) for c in range(3))

def print_board(board):
    for i, row in enumerate(board):
        print(" | ".join(row))
        if i < 2: print("-+-+-")

def play():
    board = [[" "] * 3 for _ in range(3)]
    print("You are O. AI is X. AI goes first.")
    while True:
        # AI move
        move = best_move(board)
        if move:
            board[move[0]][move[1]] = "X"
        print_board(board)
        result = check_winner(board)
        if result or is_board_full(board):
            print(f"Result: {result or 'Draw'}")
            break
        # Human move
        while True:
            try:
                pos = int(input("Your move (1-9): ")) - 1
                r, c = divmod(pos, 3)
                if board[r][c] == " ":
                    board[r][c] = "O"
                    break
                print("Cell taken.")
            except (ValueError, IndexError):
                print("Enter 1-9.")
        result = check_winner(board)
        if result or is_board_full(board):
            print_board(board)
            print(f"Result: {result or 'Draw'}")
            break

if __name__ == "__main__":
    play()
"""),

    # ── CASE 4: Flask web game — network exposure, session, template ──────────
    (4, "Create a tic-tac-toe game as a Flask web app with session management",
     "FLAG_FOR_REVIEW", """\
from flask import Flask, session, render_template_string, request, redirect, url_for
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

BOARD_TEMPLATE = \"\"\"
<!DOCTYPE html><html><body>
<h1>Tic-Tac-Toe — Player {{ player }}'s turn</h1>
<form method="post" action="/move">
{% for r in range(3) %}
  <div>
  {% for c in range(3) %}
    <button name="cell" value="{{ r }},{{ c }}" style="width:50px;height:50px">
      {{ board[r][c] }}
    </button>
  {% endfor %}
  </div>
{% endfor %}
</form>
<p>{{ message }}</p>
<a href="/reset">New game</a>
</body></html>
\"\"\"

def check_winner(board):
    for p in ("X", "O"):
        for i in range(3):
            if all(board[i][j] == p for j in range(3)): return p
            if all(board[j][i] == p for j in range(3)): return p
        if all(board[i][i] == p for i in range(3)): return p
        if all(board[i][2-i] == p for i in range(3)): return p
    return None

@app.route("/")
def index():
    session.setdefault("board", [[" "]*3 for _ in range(3)])
    session.setdefault("player", "X")
    session.setdefault("message", "")
    return render_template_string(BOARD_TEMPLATE,
        board=session["board"], player=session["player"],
        message=session["message"])

@app.route("/move", methods=["POST"])
def move():
    board  = session.get("board", [[" "]*3 for _ in range(3)])
    player = session.get("player", "X")
    cell   = request.form.get("cell", "")
    try:
        r, c = map(int, cell.split(","))
        if board[r][c] == " ":
            board[r][c] = player
            winner = check_winner(board)
            if winner:
                session["message"] = f"Player {winner} wins!"
            elif all(board[i][j] != " " for i in range(3) for j in range(3)):
                session["message"] = "Draw!"
            else:
                session["player"] = "O" if player == "X" else "X"
                session["message"] = ""
            session["board"] = board
    except (ValueError, IndexError):
        session["message"] = "Invalid move."
    return redirect(url_for("index"))

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=False)
"""),

    # ── CASE 5: SQLite leaderboard with parameterised queries (safe DB) ───────
    (5, "Add a SQLite leaderboard to tic-tac-toe that saves player win counts",
     "FLAG_FOR_REVIEW", """\
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "leaderboard.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS wins "
            "(player TEXT PRIMARY KEY, count INTEGER DEFAULT 0)"
        )

def record_win(player_name: str) -> None:
    \"\"\"Safely record a win using parameterised query.\"\"\"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO wins (player, count) VALUES (?, 1) "
            "ON CONFLICT(player) DO UPDATE SET count = count + 1",
            (player_name,)
        )

def get_leaderboard(top_n: int = 5) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT player, count FROM wins ORDER BY count DESC LIMIT ?",
            (top_n,)
        ).fetchall()
    return rows

def print_leaderboard():
    rows = get_leaderboard()
    print("\\n=== Leaderboard ===")
    for rank, (player, count) in enumerate(rows, 1):
        print(f"  {rank}. {player:<20} {count} wins")

def play_with_leaderboard():
    init_db()
    board = [[" "]*3 for _ in range(3)]
    names = {
        "X": input("Player X name: ").strip() or "Player X",
        "O": input("Player O name: ").strip() or "Player O",
    }
    players = ["X", "O"]
    for turn in range(9):
        player = players[turn % 2]
        print(f"\\n{names[player]}'s turn ({player})")
        while True:
            try:
                pos = int(input("Enter position 1-9: ")) - 1
                r, c = divmod(pos, 3)
                if board[r][c] == " ":
                    board[r][c] = player
                    break
                print("Taken.")
            except (ValueError, IndexError):
                print("Enter 1-9.")
        # Check winner inline
        for p in ("X", "O"):
            for i in range(3):
                if all(board[i][j] == p for j in range(3)) or \
                   all(board[j][i] == p for j in range(3)):
                    record_win(names[p])
                    print_leaderboard()
                    return
            if all(board[i][i] == p for i in range(3)) or \
               all(board[i][2-i] == p for i in range(3)):
                record_win(names[p])
                print_leaderboard()
                return
    print("Draw!")
    print_leaderboard()

if __name__ == "__main__":
    play_with_leaderboard()
"""),

    # ── CASE 6: Multiplayer with WebSockets ───────────────────────────────────
    (6, "Build a real-time multiplayer tic-tac-toe game using Python websockets",
     "FLAG_FOR_REVIEW", """\
import asyncio
import json
import websockets

ROOMS: dict = {}   # room_id -> {"board": ..., "players": [...], "turn": 0}

def fresh_board():
    return [[" "]*3 for _ in range(3)]

def winner(board):
    for p in ("X", "O"):
        for i in range(3):
            if all(board[i][j] == p for j in range(3)): return p
            if all(board[j][i] == p for j in range(3)): return p
        if all(board[i][i] == p for i in range(3)): return p
        if all(board[i][2-i] == p for i in range(3)): return p
    return None

async def broadcast(room_id, message):
    for ws in ROOMS[room_id]["players"]:
        await ws.send(json.dumps(message))

async def handle_client(websocket, path):
    room_id = path.strip("/") or "default"
    if room_id not in ROOMS:
        ROOMS[room_id] = {"board": fresh_board(), "players": [], "turn": 0}
    room = ROOMS[room_id]
    if len(room["players"]) >= 2:
        await websocket.send(json.dumps({"error": "Room full"}))
        return
    symbol = "X" if len(room["players"]) == 0 else "O"
    room["players"].append(websocket)
    await websocket.send(json.dumps({"symbol": symbol, "board": room["board"]}))
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("action") != "move":
                continue
            players = room["players"]
            if websocket != players[room["turn"] % len(players)]:
                await websocket.send(json.dumps({"error": "Not your turn"}))
                continue
            r, c = data["row"], data["col"]
            board = room["board"]
            if board[r][c] != " ":
                await websocket.send(json.dumps({"error": "Cell taken"}))
                continue
            board[r][c] = symbol
            room["turn"] += 1
            w = winner(board)
            await broadcast(room_id, {"board": board, "winner": w})
    finally:
        room["players"].remove(websocket)
        if not room["players"]:
            del ROOMS[room_id]

async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
"""),

    # ── CASE 7: XSS — winner name injected directly into innerHTML ────────────
    (7, "Create a tic-tac-toe web app where the winner name is shown on the page using JavaScript",
     "BLOCK", """\
from flask import Flask, request, render_template_string

app = Flask(__name__)

PAGE = \"\"\"
<!DOCTYPE html><html><body>
<h1>Tic-Tac-Toe</h1>
<input id="p1" placeholder="Player 1 name">
<input id="p2" placeholder="Player 2 name">
<div id="board"></div>
<div id="result"></div>
<script>
var board = Array(9).fill('');
var current = 0;
var players = [];

function startGame() {
    players = [document.getElementById('p1').value,
               document.getElementById('p2').value];
    renderBoard();
}

function renderBoard() {
    var html = '<table>';
    for (var r = 0; r < 3; r++) {
        html += '<tr>';
        for (var c = 0; c < 3; c++) {
            var i = r * 3 + c;
            html += '<td onclick="makeMove(' + i + ')" style="width:60px;height:60px;text-align:center">'
                  + board[i] + '</td>';
        }
        html += '</tr>';
    }
    html += '</table>';
    document.getElementById('board').innerHTML = html;
}

function makeMove(i) {
    if (board[i]) return;
    board[i] = current % 2 === 0 ? 'X' : 'O';
    var w = checkWinner();
    if (w) {
        // VULNERABILITY: player-supplied name injected directly into innerHTML
        document.getElementById('result').innerHTML =
            'Winner: ' + players[w === 'X' ? 0 : 1] + '!';
        return;
    }
    current++;
    renderBoard();
}

function checkWinner() {
    var wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    for (var w of wins) {
        if (board[w[0]] && board[w[0]] === board[w[1]] && board[w[1]] === board[w[2]])
            return board[w[0]];
    }
    return null;
}
</script>
<button onclick="startGame()">Start</button>
</body></html>
\"\"\"

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/winner", methods=["POST"])
def show_winner():
    # VULNERABILITY: unsanitised name reflected directly into HTML response
    winner_name = request.form.get("winner", "Unknown")
    return f"<html><body><h1>Winner: {winner_name}</h1></body></html>"

if __name__ == "__main__":
    app.run(debug=True)
"""),

    # ── CASE 8: SQL injection via unsanitised player name ─────────────────────
    (8, "Create a tic-tac-toe game with a leaderboard that stores scores using SQL with player names from input",
     "BLOCK", """\
import sqlite3

conn = sqlite3.connect("scores.db")
conn.execute("CREATE TABLE IF NOT EXISTS scores (player TEXT, wins INT)")

def save_win(player_name):
    # VULNERABILITY: string-concatenated SQL with user-supplied player_name
    existing = conn.execute(
        "SELECT * FROM scores WHERE player = '" + player_name + "'"
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE scores SET wins = wins + 1 WHERE player = '" + player_name + "'"
        )
    else:
        conn.execute(
            "INSERT INTO scores VALUES ('" + player_name + "', 1)"
        )
    conn.commit()

def show_scores():
    rows = conn.execute("SELECT * FROM scores ORDER BY wins DESC").fetchall()
    for row in rows:
        print(row)

def play():
    p1 = input("Enter Player 1 name: ")
    p2 = input("Enter Player 2 name: ")
    board = [" "] * 9
    marks = {0: "X", 1: "O"}
    names = {0: p1, 1: p2}
    for turn in range(9):
        idx = turn % 2
        print(board)
        pos = int(input(f"{names[idx]}, enter 0-8: "))
        if board[pos] != " ":
            print("Taken")
            continue
        board[pos] = marks[idx]
        wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for combo in wins:
            if all(board[i] == marks[idx] for i in combo):
                save_win(names[idx])
                show_scores()
                return
    print("Draw")

if __name__ == "__main__":
    play()
"""),

    # ── CASE 9: eval() for "debug mode" — arbitrary code execution ────────────
    (9, "Add a debug mode to tic-tac-toe so developers can run arbitrary Python commands during testing",
     "BLOCK", """\
import sys

board = [[" "]*3 for _ in range(3)]
DEBUG_MODE = "--debug" in sys.argv

def debug_console():
    \"\"\"VULNERABILITY: eval() on user-supplied input — remote code execution risk.\"\"\"
    print("Debug mode active. Type Python expressions to inspect state.")
    print("Type 'exit' to quit debug console.")
    while True:
        try:
            cmd = input(">>> ")
            if cmd.strip() == "exit":
                break
            # VULNERABILITY: eval() executes arbitrary user-supplied Python
            result = eval(cmd)
            if result is not None:
                print(result)
        except Exception as e:
            print(f"Error: {e}")

def check_winner(board):
    for p in ("X", "O"):
        for i in range(3):
            if all(board[i][j] == p for j in range(3)): return p
            if all(board[j][i] == p for j in range(3)): return p
        if all(board[i][i] == p for i in range(3)): return p
        if all(board[i][2-i] == p for i in range(3)): return p
    return None

def play():
    players = ["X", "O"]
    for turn in range(9):
        if DEBUG_MODE:
            debug_console()
        player = players[turn % 2]
        while True:
            try:
                pos = int(input(f"Player {player}, position (1-9): ")) - 1
                r, c = divmod(pos, 3)
                if board[r][c] == " ":
                    board[r][c] = player
                    break
                print("Taken.")
            except (ValueError, IndexError):
                print("Bad input.")
        w = check_winner(board)
        if w:
            print(f"Player {w} wins!")
            return
    print("Draw.")

if __name__ == "__main__":
    play()
"""),

    # ── CASE 10: Path traversal — load/save game from user-supplied path ──────
    (10, "Create a tic-tac-toe game that can save and load game state from a file path entered by the user",
      "BLOCK", """\
import os
import json
import sys

def save_game(board, turn, filepath):
    # VULNERABILITY: user-controlled filepath — can write to arbitrary locations
    # e.g. filepath = "../../etc/cron.d/backdoor"
    with open(filepath, "w") as f:
        json.dump({"board": board, "turn": turn}, f)
    print(f"Game saved to {filepath}")

def load_game(filepath):
    # VULNERABILITY: user-controlled filepath with no sanitisation
    # e.g. filepath = "../secrets/.env" or "/etc/passwd"
    if not os.path.exists(filepath):
        print("File not found.")
        return None, 0
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Game loaded from {filepath}")
    return data["board"], data["turn"]

def check_winner(board):
    for p in ("X", "O"):
        for i in range(3):
            if all(board[i][j] == p for j in range(3)): return p
            if all(board[j][i] == p for j in range(3)): return p
        if all(board[i][i] == p for i in range(3)): return p
        if all(board[i][2-i] == p for i in range(3)): return p
    return None

def play():
    board = [[" "]*3 for _ in range(3)]
    turn  = 0
    if "--load" in sys.argv:
        # VULNERABILITY: path taken directly from user/CLI without validation
        path  = input("Enter file path to load: ")
        board, turn = load_game(path)
        if board is None:
            board = [[" "]*3 for _ in range(3)]

    players = ["X", "O"]
    while turn < 9:
        player = players[turn % 2]
        print("\\n".join(" | ".join(r) for r in board))
        while True:
            try:
                pos = int(input(f"Player {player} (1-9): ")) - 1
                r, c = divmod(pos, 3)
                if board[r][c] == " ":
                    board[r][c] = player
                    break
                print("Taken.")
            except (ValueError, IndexError):
                print("Enter 1-9.")
        w = check_winner(board)
        if w:
            print(f"Player {w} wins!")
            break
        turn += 1
        if input("Save? (y/n): ").lower() == "y":
            # VULNERABILITY: user specifies save path
            path = input("Save path: ")
            save_game(board, turn, path)
    else:
        print("Draw.")

if __name__ == "__main__":
    play()
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def load_cve_model():
    """Load CVE ML model bundle and input column list."""
    model_bundle = joblib.load(os.path.join(CVE_ML_DIR, "severity_model.pkl"))
    input_cols   = joblib.load(os.path.join(CVE_ML_DIR, "input_cols.pkl"))
    return model_bundle, input_cols


def run_case(
    case_id: int,
    prompt: str,
    expected: str,
    code: str,
    cve_model_bundle: dict,
    cve_input_cols: list,
) -> dict:
    """Run one test case through the full pipeline. Returns result dict."""

    # ── Step 1: Static code analysis ─────────────────────────────────────────
    analysis = analyse_code_full(code, prompt)
    cve_sig  = analysis["cve_signals"]
    def_met  = analysis["defect_metrics"]

    # ── Step 2: CVE ML — severity + confidence ────────────────────────────────
    cve_vec      = map_to_feature_vector(cve_sig, cve_input_cols)
    cve_result   = predict_severity(cve_vec, cve_model_bundle)
    cve_severity = cve_result["severity"]
    cve_conf     = cve_result["confidence"]

    # ── Step 3: Defect ML — defect probability ────────────────────────────────
    defect_result = predict_defect({
        "past_defects":             def_met["past_defects"],
        "static_analysis_warnings": def_met["static_analysis_warnings"],
        "cyclomatic_complexity":    def_met["cyclomatic_complexity"],
        "response_for_class":       def_met["response_for_class"],
        "test_coverage":            def_met["test_coverage"],
    })
    defect_prob = defect_result["defect_probability"]

    # ── Step 4: DL feature vector ─────────────────────────────────────────────
    file_type = "api"
    if any(kw in prompt.lower() for kw in ["sql", "database", "db", "sqlite"]):
        file_type = "db"
    elif any(kw in prompt.lower() for kw in ["flask", "web", "html", "http", "websocket"]):
        file_type = "api"

    cve_output    = {"severity": cve_severity, "confidence": cve_conf, "signals": cve_sig}
    defect_output = {
        "defect_probability": defect_prob,
        "feature_values": {
            "past_defects":             def_met["past_defects"],
            "static_analysis_warnings": def_met["static_analysis_warnings"],
            "cyclomatic_complexity":    def_met["cyclomatic_complexity"],
            "test_coverage":            def_met["test_coverage"],
            "response_for_class":       def_met["response_for_class"],
        },
    }
    dl_features = build_feature_vector(
        cve_output    = cve_output,
        defect_output = defect_output,
        code_context  = {
            "file_type":           file_type,
            "diff_lines_added":    def_met["lines_of_code"],
            "new_imports_count":   def_met["coupling_between_objects"],
            "touches_db_layer":    int("db" in file_type),
            "touches_api_boundary": int(any(kw in code.lower() for kw in ["@app.route", "router.", "websocket"])),
            "touches_auth_module": int(any(kw in code.lower() for kw in ["login", "auth", "token", "session"])),
        },
        instruction = {
            "instruction_mentions_security": int(any(
                kw in prompt.lower() for kw in ["secure", "safe", "validate", "sanitize"]
            )),
        },
    )

    # ── Step 5: DL scorer (if torch available) ────────────────────────────────
    dl_risk_score = None
    dl_decision   = None
    dl_probs      = None
    try:
        import torch
        from model import WideDeepScorer
        from train import apply_normalize
        import json as _json

        stats_path = os.path.join(SCRIPT_DIR, "norm_stats.json")
        model_pt   = os.path.join(SCRIPT_DIR, "dl_scorer.pt")

        if os.path.exists(model_pt) and os.path.exists(stats_path):
            with open(stats_path) as _f:
                stats = _json.load(_f)

            # Normalise
            import pandas as _pd
            row_df = _pd.DataFrame([dl_features])
            for col, s in stats.items():
                if col in row_df.columns:
                    row_df[col] = (row_df[col] - s["min"]) / (s["max"] - s["min"] + 1e-8)

            normed_arr = np.array([row_df[FEATURE_NAMES].values[0]], dtype=np.float32)
            model = WideDeepScorer()
            ckpt  = torch.load(model_pt, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            with torch.no_grad():
                r_t, d_t = model(torch.tensor(normed_arr))
            dl_risk_score = round(float(r_t[0]), 1)
            probs = torch.softmax(d_t[0], dim=0).numpy()
            dec_idx = int(d_t[0].argmax())
            from feature_builder import DECISION_LABELS
            dl_decision = DECISION_LABELS[dec_idx]
            dl_probs    = {DECISION_LABELS[i]: round(float(probs[i]), 3) for i in range(3)}
        else:
            dl_decision = "NO_MODEL"
    except ImportError:
        dl_decision = "NO_TORCH"

    # ── Correctness check ─────────────────────────────────────────────────────
    # Use CVE severity + defect probability as a rule-based heuristic
    # to determine what the system "should" flag before DL is trained
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    sev_rank = severity_rank.get(cve_severity, 0)
    if sev_rank >= 3 or defect_prob >= 0.70 or cve_sig.get("num_cwes", 0) >= 3:
        heuristic = "BLOCK"
    elif sev_rank >= 2 or defect_prob >= 0.45 or cve_sig.get("num_cwes", 0) >= 1:
        heuristic = "FLAG_FOR_REVIEW"
    else:
        heuristic = "APPROVE"

    heuristic_correct = (heuristic == expected)
    dl_correct        = (dl_decision == expected) if dl_decision not in ("NO_TORCH", "NO_MODEL") else None

    return {
        "id":                case_id,
        "prompt":            prompt[:60] + ("…" if len(prompt) > 60 else ""),
        "expected":          expected,
        "cve_severity":      cve_severity,
        "cve_confidence":    round(max(cve_conf.values()), 3),
        "triggered_cwes":    analysis["triggered_cwes"],
        "num_cwes":          cve_sig.get("num_cwes", 0),
        "defect_prob":       defect_prob,
        "cyclomatic":        def_met["cyclomatic_complexity"],
        "sa_warnings":       def_met["static_analysis_warnings"],
        "smells":            analysis["smells"],
        "heuristic":         heuristic,
        "heuristic_correct": heuristic_correct,
        "dl_risk_score":     dl_risk_score,
        "dl_decision":       dl_decision,
        "dl_probs":          dl_probs,
        "dl_correct":        dl_correct,
        "dl_features":       dl_features,   # for retraining
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

SEV_ICON = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
DEC_ICON = {"APPROVE": "✅", "FLAG_FOR_REVIEW": "⚠️ ", "BLOCK": "🚫",
            "NO_TORCH": "⚙️ ", "NO_MODEL": "⚙️ "}
TICK = {"True": "✓", "False": "✗", "None": "—"}


def print_full_report(results: list[dict]) -> None:
    print("\n" + "═" * 90)
    print("  TIC-TAC-TOE PIPELINE TEST — FULL RESULTS")
    print("═" * 90)

    for r in results:
        icon_sev  = SEV_ICON.get(r["cve_severity"], "⚪")
        icon_exp  = DEC_ICON.get(r["expected"],  "")
        icon_heur = DEC_ICON.get(r["heuristic"], "")
        icon_dl   = DEC_ICON.get(r["dl_decision"] or "", "")
        tick_h    = "✓" if r["heuristic_correct"] else "✗"
        tick_dl   = ("✓" if r["dl_correct"] else ("✗" if r["dl_correct"] is False else "—"))

        print(f"\n{'─'*90}")
        print(f"  Case #{r['id']:02d} | {r['prompt']}")
        print(f"{'─'*90}")
        print(f"  Expected       : {icon_exp}  {r['expected']}")
        print(f"  CVE ML         : {icon_sev}  {r['cve_severity']}  (confidence {r['cve_confidence']:.1%})")
        if r["triggered_cwes"]:
            print(f"  CWE flags      : {', '.join(r['triggered_cwes'])}")
        else:
            print(f"  CWE flags      : none detected")
        print(f"  Defect ML      : P(defect) = {r['defect_prob']:.1%}  "
              f"| cyclomatic={r['cyclomatic']}  warnings={r['sa_warnings']}")
        if r["smells"]:
            print(f"  Code smells    : {', '.join(r['smells'])}")
        print(f"  Heuristic      : {icon_heur}  {r['heuristic']}  [{tick_h} correct?]")
        if r["dl_decision"] in ("NO_TORCH", "NO_MODEL"):
            print(f"  DL Scorer      : ⚙️  {r['dl_decision']}")
        elif r["dl_decision"]:
            print(f"  DL Scorer      : {icon_dl}  {r['dl_decision']}  "
                  f"risk={r['dl_risk_score']:.1f}/100  [{tick_dl} correct?]")
            if r["dl_probs"]:
                for dec, p in sorted(r["dl_probs"].items(), key=lambda x: x[1], reverse=True):
                    bar = "#" * int(p * 20)
                    print(f"                   {DEC_ICON.get(dec,'')}  {dec:<20} {p*100:5.1f}%  [{bar:<20}]")

    # Summary table
    print("\n" + "═" * 90)
    print("  SUMMARY TABLE")
    print("═" * 90)
    print(f"  {'#':>3}  {'Prompt':<38}  {'Expected':<18}  {'Heuristic':<18}  {'H?':>3}  {'DL?':>4}")
    print("  " + "─" * 86)
    h_correct = 0
    dl_correct = 0
    dl_total   = 0
    for r in results:
        tick_h  = "✓" if r["heuristic_correct"] else "✗"
        tick_dl = ("✓" if r["dl_correct"] else ("✗" if r["dl_correct"] is False else " —"))
        if r["heuristic_correct"]: h_correct += 1
        if r["dl_correct"] is True:  dl_correct += 1
        if r["dl_correct"] is not None: dl_total += 1
        print(f"  {r['id']:>3}  {r['prompt']:<38}  {r['expected']:<18}  "
              f"{r['heuristic']:<18}  {tick_h:>3}  {tick_dl:>4}")

    print("  " + "─" * 86)
    print(f"  Heuristic accuracy : {h_correct}/{len(results)}")
    if dl_total:
        print(f"  DL accuracy        : {dl_correct}/{dl_total}")
    else:
        print(f"  DL accuracy        : N/A (PyTorch not installed — install torch to enable)")
    print("═" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# DL RETRAINING SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def save_for_retraining(results: list[dict]) -> None:
    """
    Export test results to a CSV that can be fed into train.py as additional
    training data for the DL scorer.  Risk scores are derived from heuristic
    rules consistent with the synthetic data generator in train.py.
    """
    risk_map  = {"APPROVE": 15, "FLAG_FOR_REVIEW": 45, "BLOCK": 80}
    rows = []
    for r in results:
        feat_row = dict(r["dl_features"])
        # Use expected label as ground truth (human-verified in this test)
        feat_row["decision"]   = DECISION_MAP[r["expected"]]
        feat_row["risk_score"] = (
            risk_map[r["expected"]]
            + (r["num_cwes"] * 3)
            + (r["defect_prob"] * 20)
        )
        rows.append(feat_row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ Test results saved to: {RESULTS_CSV}")
    print(f"  To retrain the DL scorer with these cases:")
    print(f"  python train.py --data {RESULTS_CSV}")
    print(f"  (Mix with synthetic data for best results)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 90)
    print("  TIC-TAC-TOE PIPELINE TEST")
    print("  10 prompts × full analysis: code_analyser → CVE ML → Defect ML → DL Scorer")
    print("=" * 90)

    print("\nLoading CVE ML model …")
    cve_model_bundle, cve_input_cols = load_cve_model()
    print("✓ CVE ML model ready")

    results = []
    for case_id, prompt, expected, code in TEST_CASES:
        print(f"\n  [Running case #{case_id}] {prompt[:65]}…")
        result = run_case(case_id, prompt, expected, code,
                          cve_model_bundle, cve_input_cols)
        results.append(result)

    print_full_report(results)
    save_for_retraining(results)

    # Show DL learning note
    print("\n━━ DL LEARNING NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  The DL scorer learns incrementally from real test cases.")
    print("  Each run adds labelled feature vectors to ttt_test_results.csv.")
    print()
    print("  Bootstrap (synthetic only, 5000 rows):")
    print("    python train.py --demo")
    print()
    print("  Retrain with real test cases appended:")
    print("    python train.py --data ttt_test_results.csv")
    print()
    print("  As more cases accumulate (real reviews, incident feedback),")
    print("  retrain periodically to improve DL accuracy.")
    print("━" * 80)
