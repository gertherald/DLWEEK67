"""
demo_20_cases.py — 80-Case Learning Demo with 4 Retrain Cycles
===============================================================
Demonstrates the full auto-retrain learning cycle across four rounds:

  Phase 0  Bootstrap model — 5,000 synthetic rows (Tier-2 weight =  0%)
  Phase 1  Log 20 Round-1 reviews → retrain #1 (Tier-2 weight = 20%)
  Phase 2  Log 20 Round-2 reviews → retrain #2 (Tier-2 weight = 40%)
  Phase 3  Log 20 Round-3 reviews → retrain #3 (Tier-2 weight = 60%)
  Phase 4  Log 20 Round-4 reviews → retrain #4 (Tier-2 weight = 80%)
  Phase 5  Score 10 validation cases with all five models
  Phase 6  5-column comparison tables (Bootstrap | R1 | R2 | R3 | R4)
  Phase 7  Summary — accuracy progression + reasoning evolution

Dynamic feature weighting:
  As real reviews accumulate, user-feedback features gain signal:
    0 reviews  → Tier-2 weight = 0%   (shadow_twin≈0, sentiment≈0)
   20 reviews  → Tier-2 weight = 20%  (partial shadow signal emerges)
   40 reviews  → Tier-2 weight = 40%  (user feedback starts shaping labels)
   60 reviews  → Tier-2 weight = 60%  (user accuracy and override patterns)
   80 reviews  → Tier-2 weight = 80%  (near-full user feedback signal)
  100 reviews  → Tier-2 weight = 100% (full user feedback)

Per-case reasoning shows which feature tier drove each decision and
how the reasoning explanation evolves across all four retrain cycles.

Run:
  cd "src/DL SCORER"
  python demo_20_cases.py
"""

import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CVE_ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "CVE ML"))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if CVE_ML_DIR not in sys.path:
    sys.path.insert(0, CVE_ML_DIR)

from code_analyser     import analyse_code_full
from defect_scorer     import predict_defect
from feature_builder   import build_feature_vector, FEATURE_NAMES, DECISION_MAP, DECISION_LABELS
from codex_risk_pipeline import predict_severity, map_to_feature_vector
from auto_retrain      import (
    generate_weighted_synthetic_data, generate_synthetic_data,
    fit_normalize, apply_normalize,
)
from demo_learning     import train_mlp
from review_logger     import log_review
from scorer            import score_commit
from test_tic_tac_toe  import TEST_CASES, load_cve_model
from shadow_runner     import run_shadow_twin, warm_up as _shadow_warm_up

# ── Display helpers ────────────────────────────────────────────────────────────
DEC_ICON  = {"APPROVE": "✅", "FLAG_FOR_REVIEW": "⚠️ ", "BLOCK": "🚫"}
TICK_OK   = "✓"
TICK_BAD  = "✗"


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ROUND 1 — 10 NEW CASES (cases 11–20, combined with TEST_CASES 1-10)
# ═══════════════════════════════════════════════════════════════════════════════

NEW_TRAINING_CASES: list[tuple] = [

    # ── CASE 11: OOP class-based design (APPROVE) ─────────────────────────────
    (11, "Implement tic-tac-toe using OOP with a Board class and a Game class",
     "APPROVE", """\
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
"""),

    # ── CASE 12: NumPy board with pretty-print (APPROVE) ──────────────────────
    (12, "Write tic-tac-toe that stores the board as a NumPy array",
     "APPROVE", """\
import numpy as np

def create_board() -> np.ndarray:
    return np.full((3, 3), ' ')

def display_board(board: np.ndarray) -> None:
    for i, row in enumerate(board):
        print(' | '.join(row))
        if i < 2:
            print('---------')

def check_winner(board: np.ndarray, player: str) -> bool:
    if np.any(np.all(board == player, axis=0)): return True
    if np.any(np.all(board == player, axis=1)): return True
    if np.all(np.diag(board) == player):        return True
    if np.all(np.diag(np.fliplr(board)) == player): return True
    return False

def play() -> None:
    board   = create_board()
    players = ['X', 'O']
    for turn in range(9):
        player = players[turn % 2]
        display_board(board)
        try:
            row = int(input(f'Player {player} row (0-2): '))
            col = int(input(f'Player {player} col (0-2): '))
        except ValueError:
            print('Enter a number.')
            continue
        if not (0 <= row <= 2 and 0 <= col <= 2) or board[row, col] != ' ':
            print('Invalid move.')
            continue
        board[row, col] = player
        if check_winner(board, player):
            display_board(board)
            print(f'Player {player} wins!')
            return
    print('Draw!')

if __name__ == '__main__':
    play()
"""),

    # ── CASE 13: Pytest unit-tested game logic (APPROVE) ──────────────────────
    (13, "Write tic-tac-toe game logic with full pytest unit test coverage",
     "APPROVE", """\
def check_winner(board: list, player: str) -> bool:
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    return any(all(board[i] == player for i in line) for line in lines)

def make_move(board: list, pos: int, player: str) -> bool:
    if 0 <= pos <= 8 and board[pos] == ' ':
        board[pos] = player
        return True
    return False

def is_draw(board: list) -> bool:
    return ' ' not in board and \
           not check_winner(board, 'X') and not check_winner(board, 'O')

def test_row_win():
    board = ['X','X','X','O','O',' ',' ',' ',' ']
    assert check_winner(board, 'X') is True
    assert check_winner(board, 'O') is False

def test_col_win():
    board = ['O','X',' ','O','X',' ','O',' ','X']
    assert check_winner(board, 'O') is True

def test_diagonal_win():
    board = ['X','O','O',' ','X','O',' ',' ','X']
    assert check_winner(board, 'X') is True

def test_draw():
    board = ['X','O','X','O','X','O','O','X','O']
    assert is_draw(board) is True
"""),

    # ── CASE 14: Async I/O game loop (APPROVE) ────────────────────────────────
    (14, "Write a tic-tac-toe game using Python asyncio for non-blocking I/O",
     "APPROVE", """\
import asyncio

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

async def get_move_async(player: str) -> int:
    loop = asyncio.get_event_loop()
    while True:
        raw = await loop.run_in_executor(
            None, lambda: input(f'Player {player} enter position (0-8): ')
        )
        try:
            pos = int(raw)
            if 0 <= pos <= 8:
                return pos
        except ValueError:
            pass
        print('Please enter a number 0-8.')

async def play_async() -> None:
    board   = [' '] * 9
    players = ['X', 'O']

    for turn in range(9):
        player = players[turn % 2]
        print('\\n'.join(' | '.join(board[i:i+3]) for i in range(0, 9, 3)))

        while True:
            pos = await get_move_async(player)
            if board[pos] == ' ':
                break
            print('That cell is taken. Try again.')

        board[pos] = player
        if check_winner(board, player):
            print(f'\\nPlayer {player} wins!')
            return

    print('\\nDraw!')

if __name__ == '__main__':
    asyncio.run(play_async())
"""),

    # ── CASE 15: HTTP server with API-key header (FLAG) ───────────────────────
    (15, "Create a tic-tac-toe REST API server with API key authentication",
     "FLAG_FOR_REVIEW", """\
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
"""),

    # ── CASE 16: Redis leaderboard (FLAG) ─────────────────────────────────────
    (16, "Add a Redis-backed leaderboard to tic-tac-toe that persists win counts",
     "FLAG_FOR_REVIEW", """\
import redis

_redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def record_win(player_name: str) -> None:
    _redis.zincrby('ttt:leaderboard', 1, player_name)

def get_leaderboard(n: int = 5) -> list:
    return _redis.zrevrange('ttt:leaderboard', 0, n - 1, withscores=True)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    board   = [' '] * 9
    players = ['X', 'O']
    names   = [input('Player X name: '), input('Player O name: ')]
    for turn in range(9):
        sym  = players[turn % 2]
        name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}) position (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = sym
        if check_winner(board, sym):
            record_win(name)
            print(f'{name} wins!  Leaderboard: {get_leaderboard()}')
            return
    print('Draw!')
"""),

    # ── CASE 17: Config file with player settings (FLAG) ──────────────────────
    (17, "Create a tic-tac-toe game that reads player settings from a config file",
     "FLAG_FOR_REVIEW", """\
import json
import os

CONFIG_PATH    = 'ttt_settings.json'
DEFAULT_CONFIG = {
    'player1_symbol': 'X',
    'player2_symbol': 'O',
    'show_hints':     True,
    'board_size':     3,
}
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG

def check_winner(board: list, mark: str) -> bool:
    return any(all(board[i] == mark for i in line) for line in WIN_LINES)

def play(config: dict = None) -> None:
    cfg    = config or load_config()
    syms   = [cfg['player1_symbol'], cfg['player2_symbol']]
    board  = [' '] * 9
    for turn in range(9):
        sym = syms[turn % 2]
        try:
            pos = int(input(f'Player {sym} move (0-8): '))
        except ValueError:
            continue
        if board[pos] == ' ':
            board[pos] = sym
            if check_winner(board, sym):
                print(f'{sym} wins!')
                return
    print('Draw!')
"""),

    # ── CASE 18: subprocess + user-controlled command (BLOCK) ─────────────────
    (18, "Make tic-tac-toe log each move to a file using a shell command",
     "BLOCK", """\
import subprocess

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def log_move(player_name: str, position: int) -> None:
    # Command injection: player_name is unsanitised
    cmd = f'echo "Player {player_name} moved to {position}" >> ttt_moves.log'
    subprocess.call(cmd, shell=True)

def announce_winner(winner_name: str) -> None:
    # Injection point: winner name from user input used in shell command
    subprocess.call(f'say "Congratulations {winner_name}"', shell=True)

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
        log_move(name, pos)      # ← injection
        if check_winner(board, sym):
            announce_winner(name)   # ← injection
            return
    print('Draw!')
"""),

    # ── CASE 19: pickle.loads game state (BLOCK) ──────────────────────────────
    (19, "Add save/load game state to tic-tac-toe using pickle serialization",
     "BLOCK", """\
import pickle
import base64

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_state(board: list, turn: int) -> str:
    data = {'board': board, 'turn': turn}
    return base64.b64encode(pickle.dumps(data)).decode()

def load_state(encoded: str) -> dict:
    # Unsafe: pickle.loads executes arbitrary Python bytecode
    return pickle.loads(base64.b64decode(encoded))

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    resume = input('Paste saved game state (or press Enter for new game): ').strip()
    if resume:
        state = load_state(resume)     # ← unsafe deserialization
        board = state['board']
        turn  = state['turn']
    else:
        board = [' '] * 9
        turn  = 0

    players = ['X', 'O']
    while turn < 9:
        player = players[turn % 2]
        try:
            pos = int(input(f'Player {player} (0-8): '))
        except ValueError:
            continue
        board[pos] = player
        turn += 1
        if check_winner(board, player):
            print(f'{player} wins!')
            print(f'Save state: {save_state(board, turn)}')
            return
    print('Draw!')
"""),

    # ── CASE 20: Hardcoded creds + SQL injection (BLOCK) ──────────────────────
    (20, "Write tic-tac-toe that stores game results in a SQLite database",
     "BLOCK", """\
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
            save_result(name)     # ← SQL injection
            print(f'{name} wins!')
            return
    print('Draw!')
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ROUND 2 — 20 NEW CASES (21–40)
# Different code, same vulnerability patterns — reinforces learning from R1
# APPROVE (7) · FLAG (6) · BLOCK (7)
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CASES_2: list[tuple] = [

    # ── CASE 21: Dataclass game state (APPROVE) ───────────────────────────────
    (21, "Use Python dataclasses to represent tic-tac-toe game state cleanly",
     "APPROVE", """\
from dataclasses import dataclass, field
from typing import Optional

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

@dataclass
class GameState:
    board:   list  = field(default_factory=lambda: [' '] * 9)
    turn:    int   = 0
    winner:  Optional[str] = None

    @property
    def current_player(self) -> str:
        return 'X' if self.turn % 2 == 0 else 'O'

    def move(self, pos: int) -> bool:
        if not (0 <= pos <= 8) or self.board[pos] != ' ':
            return False
        self.board[pos] = self.current_player
        if any(all(self.board[i] == self.current_player for i in ln) for ln in WIN_LINES):
            self.winner = self.current_player
        self.turn += 1
        return True

    def display(self) -> None:
        for i in range(0, 9, 3):
            print(' | '.join(self.board[i:i+3]))
            if i < 6: print('---------')


def play() -> None:
    g = GameState()
    while g.winner is None and g.turn < 9:
        g.display()
        try:
            pos = int(input(f'Player {g.current_player} (0-8): '))
        except ValueError:
            continue
        if not g.move(pos):
            print('Invalid.')
    g.display()
    print(f'{g.winner} wins!' if g.winner else 'Draw!')
"""),

    # ── CASE 22: Immutable tuple-based board (APPROVE) ────────────────────────
    (22, "Implement tic-tac-toe with immutable state using tuples and recursion",
     "APPROVE", """\
from typing import Optional, Tuple

WIN_LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

Board = Tuple[str, ...]

def new_board() -> Board:
    return (' ',) * 9

def check_winner(board: Board) -> Optional[str]:
    for p in ('X', 'O'):
        if any(all(board[i] == p for i in ln) for ln in WIN_LINES):
            return p
    return None

def make_move(board: Board, pos: int, player: str) -> Board:
    if not (0 <= pos <= 8) or board[pos] != ' ':
        raise ValueError('Illegal move')
    return board[:pos] + (player,) + board[pos+1:]

def display(board: Board) -> None:
    for i in range(0, 9, 3):
        print(' | '.join(board[i:i+3]))
        if i < 6: print('---------')

def play(board: Board = new_board(), turn: int = 0) -> None:
    display(board)
    winner = check_winner(board)
    if winner:
        print(f'{winner} wins!')
        return
    if turn == 9:
        print('Draw!')
        return
    player = 'X' if turn % 2 == 0 else 'O'
    try:
        pos = int(input(f'Player {player} (0-8): '))
        play(make_move(board, pos, player), turn + 1)
    except (ValueError, IndexError):
        print('Invalid. Try again.')
        play(board, turn)
"""),

    # ── CASE 23: Rich terminal UI with colors (APPROVE) ───────────────────────
    (23, "Add colorful ANSI terminal output to tic-tac-toe for a better UI",
     "APPROVE", """\
import sys

RESET = '\\033[0m'
RED   = '\\033[31m'
BLUE  = '\\033[34m'
BOLD  = '\\033[1m'
CLEAR = '\\033[2J\\033[H'

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def colour(mark: str) -> str:
    if mark == 'X': return f'{RED}{BOLD}X{RESET}'
    if mark == 'O': return f'{BLUE}{BOLD}O{RESET}'
    return f' '

def display(board: list) -> None:
    print(CLEAR)
    rows = [[colour(board[i+j]) for j in range(3)] for i in range(0, 9, 3)]
    for k, row in enumerate(rows):
        print(' ' + ' | '.join(row))
        if k < 2:
            print('---+---+---')

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        display(board)
        try:
            pos = int(input(f'  Player {p} — enter position (0-8): '))
            if not (0 <= pos <= 8) or board[pos] != ' ':
                print('Invalid move.')
                continue
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            display(board)
            print(f'  {colour(p)} wins!')
            return
    display(board)
    print("  It's a draw!")
"""),

    # ── CASE 24: Position-dict board (APPROVE) ────────────────────────────────
    (24, "Store the tic-tac-toe board as a dictionary mapping position to mark",
     "APPROVE", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def new_board() -> dict:
    return {i: ' ' for i in range(9)}

def display(board: dict) -> None:
    for row_start in range(0, 9, 3):
        print(' | '.join(board[row_start + c] for c in range(3)))
        if row_start < 6:
            print('- + - + -')

def check_winner(board: dict, mark: str) -> bool:
    return any(all(board[i] == mark for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = new_board()
    players = ['X', 'O']
    taken   = set()
    for turn in range(9):
        p = players[turn % 2]
        display(board)
        try:
            pos = int(input(f'Player {p} choose position 0-8: '))
        except ValueError:
            print('Numbers only.')
            continue
        if pos not in board or pos in taken:
            print('Invalid or taken.')
            continue
        board[pos] = p
        taken.add(pos)
        if check_winner(board, p):
            display(board)
            print(f'Player {p} wins!')
            return
    display(board)
    print("Draw!")
"""),

    # ── CASE 25: Alpha-beta minimax AI (APPROVE) ──────────────────────────────
    (25, "Implement an unbeatable tic-tac-toe AI using alpha-beta pruning minimax",
     "APPROVE", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def winner(board):
    for p in ('X','O'):
        if any(all(board[i]==p for i in ln) for ln in WIN_LINES):
            return p
    return 'D' if ' ' not in board else None

def minimax(board, is_max, alpha=-10, beta=10):
    w = winner(board)
    if w == 'O': return  1
    if w == 'X': return -1
    if w == 'D': return  0
    best = -10 if is_max else 10
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O' if is_max else 'X'
            score = minimax(board, not is_max, alpha, beta)
            board[i] = ' '
            if is_max:
                best  = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if beta <= alpha:
                break
    return best

def best_move(board):
    return max(
        (i for i in range(9) if board[i]==' '),
        key=lambda i: (board.__setitem__(i,'O') or minimax(board[:i]+[' ']+board[i+1:],False))
    )

def play():
    board = [' ']*9
    for turn in range(9):
        if turn % 2 == 0:
            try:
                pos = int(input('Your move (0-8): '))
                board[pos] = 'X'
            except (ValueError, IndexError):
                continue
        else:
            pos = best_move(board)
            board[pos] = 'O'
            print(f'AI plays {pos}')
        print('\\n'.join(' | '.join(board[r:r+3]) for r in range(0,9,3)))
        w = winner(board)
        if w:
            print(f'{w} wins!' if w != 'D' else 'Draw!')
            return
"""),

    # ── CASE 26: Enum-based marks (APPROVE) ───────────────────────────────────
    (26, "Use Python Enum to represent player marks and game outcomes clearly",
     "APPROVE", """\
from enum import Enum, auto
from typing import Optional

class Mark(Enum):
    X     = 'X'
    O     = 'O'
    EMPTY = ' '

class Outcome(Enum):
    X_WINS   = auto()
    O_WINS   = auto()
    DRAW     = auto()
    ONGOING  = auto()

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_outcome(board: list) -> Outcome:
    for m in (Mark.X, Mark.O):
        if any(all(board[i] == m for i in ln) for ln in WIN_LINES):
            return Outcome.X_WINS if m == Mark.X else Outcome.O_WINS
    if all(c != Mark.EMPTY for c in board):
        return Outcome.DRAW
    return Outcome.ONGOING

def play() -> None:
    board   = [Mark.EMPTY] * 9
    players = [Mark.X, Mark.O]
    for turn in range(9):
        m = players[turn % 2]
        print('\\n'.join(' | '.join(c.value for c in board[i:i+3]) for i in range(0,9,3)))
        try:
            pos = int(input(f'Player {m.value} (0-8): '))
            if board[pos] != Mark.EMPTY:
                print('Taken.')
                continue
        except (ValueError, IndexError):
            continue
        board[pos] = m
        out = check_outcome(board)
        if out != Outcome.ONGOING:
            print(out.name.replace('_', ' '))
            return
"""),

    # ── CASE 27: Logging + standard lib only (APPROVE) ────────────────────────
    (27, "Add Python standard logging to tic-tac-toe to track game events",
     "APPROVE", """\
import logging

logging.basicConfig(
    filename='ttt_game.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger('ttt')

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = [' '] * 9
    players = ['X', 'O']
    logger.info('New game started')
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
            if not (0 <= pos <= 8) or board[pos] != ' ':
                logger.warning('Player %s invalid move %d', p, pos)
                continue
        except ValueError:
            continue
        board[pos] = p
        logger.info('Player %s placed at %d', p, pos)
        if check_winner(board, p):
            logger.info('Player %s wins', p)
            print(f'{p} wins!')
            return
    logger.info('Game ended in draw')
    print('Draw!')
"""),

    # ── CASE 28: Django template game view (FLAG) ─────────────────────────────
    (28, "Build a Django view and template for a browser-based tic-tac-toe game",
     "FLAG_FOR_REVIEW", """\
# views.py
from django.shortcuts import render
from django.views import View
from django.contrib.sessions.backends.db import SessionStore

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def _check_winner(board, player):
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

class GameView(View):
    def get(self, request):
        board = request.session.get('board', [' '] * 9)
        turn  = request.session.get('turn', 0)
        return render(request, 'ttt/board.html', {
            'board': board, 'turn': turn,
            'player': 'X' if turn % 2 == 0 else 'O',
        })

    def post(self, request):
        board = request.session.get('board', [' '] * 9)
        turn  = request.session.get('turn', 0)
        pos   = int(request.POST.get('position', -1))
        player = 'X' if turn % 2 == 0 else 'O'
        if 0 <= pos <= 8 and board[pos] == ' ':
            board[pos] = player
            turn += 1
        request.session['board'] = board
        request.session['turn']  = turn
        winner = player if _check_winner(board, player) else None
        return render(request, 'ttt/board.html', {
            'board': board, 'winner': winner, 'turn': turn,
        })
"""),

    # ── CASE 29: FastAPI + Pydantic (FLAG) ────────────────────────────────────
    (29, "Expose tic-tac-toe as a FastAPI REST service with Pydantic validation",
     "FLAG_FOR_REVIEW", """\
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app   = FastAPI(title="Tic-Tac-Toe API")
board = [' '] * 9
turn  = 0

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class MoveRequest(BaseModel):
    position: int = Field(..., ge=0, le=8)
    player:   str = Field(..., pattern='^[XO]$')

class GameState(BaseModel):
    board:  list
    turn:   int
    winner: Optional[str] = None

@app.get('/state', response_model=GameState)
def get_state():
    return GameState(board=board, turn=turn)

@app.post('/move', response_model=GameState)
def make_move(req: MoveRequest):
    global board, turn
    if board[req.position] != ' ':
        raise HTTPException(status_code=400, detail='Cell taken')
    board[req.position] = req.player
    turn += 1
    winner = req.player if any(
        all(board[i] == req.player for i in ln) for ln in WIN_LINES
    ) else None
    return GameState(board=board, turn=turn, winner=winner)
"""),

    # ── CASE 30: MongoDB win tracker (FLAG) ───────────────────────────────────
    (30, "Track tic-tac-toe player wins in a MongoDB collection",
     "FLAG_FOR_REVIEW", """\
from pymongo import MongoClient

_client = MongoClient('mongodb://localhost:27017/')
_db     = _client['tictactoe']
_scores = _db['scores']

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def record_win(name: str) -> None:
    _scores.update_one({'player': name}, {'$inc': {'wins': 1}}, upsert=True)

def leaderboard(top_n: int = 5) -> list:
    return list(
        _scores.find({}, {'_id': 0, 'player': 1, 'wins': 1})
                .sort('wins', -1).limit(top_n)
    )

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
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = sym
        if check_winner(board, sym):
            record_win(name)
            print(f'{name} wins!  Top players: {leaderboard()}')
            return
    print('Draw!')
"""),

    # ── CASE 31: WebSocket multiplayer (FLAG) ─────────────────────────────────
    (31, "Implement real-time tic-tac-toe multiplayer using WebSockets",
     "FLAG_FOR_REVIEW", """\
import asyncio
import json
import websockets

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
_state   = {'board': [' ']*9, 'turn': 0}
_clients = set()

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

async def broadcast(msg: dict) -> None:
    if _clients:
        await asyncio.gather(*[c.send(json.dumps(msg)) for c in _clients])

async def handler(ws) -> None:
    _clients.add(ws)
    await ws.send(json.dumps({'type': 'state', 'state': _state}))
    try:
        async for raw in ws:
            msg  = json.loads(raw)
            pos  = int(msg.get('position', -1))
            player = 'X' if _state['turn'] % 2 == 0 else 'O'
            if 0 <= pos <= 8 and _state['board'][pos] == ' ':
                _state['board'][pos] = player
                _state['turn'] += 1
                won = check_winner(_state['board'], player)
                await broadcast({'type': 'state', 'state': _state, 'winner': player if won else None})
    finally:
        _clients.discard(ws)

async def main():
    async with websockets.serve(handler, '0.0.0.0', 8765):
        await asyncio.Future()

asyncio.run(main())
"""),

    # ── CASE 32: SMTP email on win (FLAG) ────────────────────────────────────
    (32, "Send an email notification when a tic-tac-toe player wins the game",
     "FLAG_FOR_REVIEW", """\
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = 'smtp.example.com'
SMTP_PORT = 587
FROM_ADDR = 'ttt@example.com'

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def send_win_email(to_addr: str, winner: str) -> None:
    msg = MIMEText(f'Congratulations {winner}, you won tic-tac-toe!')
    msg['Subject'] = 'You won!'
    msg['From']    = FROM_ADDR
    msg['To']      = to_addr
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.send_message(msg)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    names  = [input('Player X name: '), input('Player O name: ')]
    emails = [input('Player X email: '), input('Player O email: ')]
    board  = [' '] * 9
    syms   = ['X', 'O']
    for turn in range(9):
        sym  = syms[turn % 2]
        try:
            pos = int(input(f'{names[turn%2]} ({sym}): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            send_win_email(emails[turn%2], names[turn%2])   # network I/O
            print(f'{names[turn%2]} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 33: Jinja2 SSTI template injection (BLOCK) ───────────────────────
    (33, "Let players customise the winner announcement with a Jinja2 template",
     "BLOCK", """\
from jinja2 import Template

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def render_message(user_template: str, winner: str, board: list) -> str:
    # SSTI: user-controlled template string is rendered directly
    tpl = Template(user_template)
    return tpl.render(winner=winner, board=board)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    # Player can inject {{config}} or {{''.__class__.__mro__[1].__subclasses__()}}
    custom_msg = input('Custom winner message (use {{winner}} for name): ')
    board      = [' '] * 9
    players    = ['X', 'O']
    for turn in range(9):
        player = players[turn % 2]
        try:
            pos = int(input(f'Player {player} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = player
        if check_winner(board, player):
            msg = render_message(custom_msg, player, board)  # ← SSTI
            print(msg)
            return
    print('Draw!')
"""),

    # ── CASE 34: os.system command injection (BLOCK) ──────────────────────────
    (34, "Make tic-tac-toe announce the winner via a system notification command",
     "BLOCK", """\
import os

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def broadcast_win(player_name: str) -> None:
    # Command injection: player_name is directly interpolated into shell command
    os.system('echo "Winner: ' + player_name + '" | wall')

def log_result(p1: str, p2: str, winner: str) -> None:
    os.system(f'echo "{p1} vs {p2}: {winner} wins" >> /tmp/ttt_log.txt')

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    p1 = input('Player X name: ')
    p2 = input('Player O name: ')
    board   = [' '] * 9
    symbols = ['X', 'O']
    names   = [p1, p2]
    for turn in range(9):
        sym  = symbols[turn % 2]
        name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}) move (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = sym
        if check_winner(board, sym):
            broadcast_win(name)        # ← injection
            log_result(p1, p2, name)   # ← injection
            return
    print('Draw!')
"""),

    # ── CASE 35: yaml.load unsafe deserialise (BLOCK) ─────────────────────────
    (35, "Allow players to paste YAML to configure tic-tac-toe settings",
     "BLOCK", """\
import yaml

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def load_settings_from_user() -> dict:
    print('Paste YAML settings (end with blank line):')
    lines = []
    while True:
        line = input()
        if line == '':
            break
        lines.append(line)
    raw = '\\n'.join(lines)
    # Unsafe: yaml.Loader allows arbitrary Python object construction
    return yaml.load(raw, Loader=yaml.Loader)

def restore_game(yaml_str: str) -> dict:
    return yaml.load(yaml_str, Loader=yaml.Loader)  # unsafe

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    settings = load_settings_from_user()  # ← unsafe deserialization
    p1_sym   = settings.get('player1', 'X')
    p2_sym   = settings.get('player2', 'O')
    board    = [' '] * 9
    syms     = [p1_sym, p2_sym]
    for turn in range(9):
        sym = syms[turn % 2]
        try:
            pos = int(input(f'{sym} move (0-8): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            print(f'{sym} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 36: XML external entity injection (BLOCK) ────────────────────────
    (36, "Save and load tic-tac-toe game state as XML with player-supplied data",
     "BLOCK", """\
import xml.etree.ElementTree as ET

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_to_xml(board: list, turn: int) -> str:
    root = ET.Element('game')
    ET.SubElement(root, 'turn').text  = str(turn)
    brd  = ET.SubElement(root, 'board')
    for i, cell in enumerate(board):
        ET.SubElement(brd, 'cell', pos=str(i)).text = cell
    return ET.tostring(root, encoding='unicode')

def load_from_xml(xml_str: str) -> dict:
    # Unsafe: defusedxml not used — vulnerable to XXE (file read / SSRF)
    root  = ET.fromstring(xml_str)
    turn  = int(root.findtext('turn', '0'))
    cells = root.find('board')
    board = [cells.find(f".//cell[@pos='{i}']").text for i in range(9)]
    return {'board': board, 'turn': turn}

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    raw = input('Paste saved XML state (or Enter for new game): ').strip()
    if raw:
        state = load_from_xml(raw)   # ← XXE
        board, turn = state['board'], state['turn']
    else:
        board, turn = [' '] * 9, 0
    players = ['X', 'O']
    while turn < 9:
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        turn += 1
        if check_winner(board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 37: marshal.loads unsafe deserialization (BLOCK) ─────────────────
    (37, "Restore a tic-tac-toe game session using Python marshal deserialization",
     "BLOCK", """\
import marshal
import base64

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_session(board: list, turn: int) -> str:
    return base64.b64encode(marshal.dumps({'board': board, 'turn': turn})).decode()

def restore_session(token: str) -> dict:
    # Unsafe: marshal.loads can execute arbitrary code objects
    raw = base64.b64decode(token)
    return marshal.loads(raw)    # ← dangerous if token comes from user

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    token = input('Paste session token (or Enter for new game): ').strip()
    if token:
        state = restore_session(token)   # ← unsafe deserialization
        board, turn = state['board'], state['turn']
    else:
        board, turn = [' ']*9, 0
    players = ['X', 'O']
    while turn < 9:
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        turn += 1
        if check_winner(board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 38: LDAP injection via concatenated filter (BLOCK) ───────────────
    (38, "Look up player profiles from an LDAP directory before starting the game",
     "BLOCK", """\
from ldap3 import Server, Connection, ALL

LDAP_SERVER = 'ldap://ldap.example.com'
BASE_DN     = 'ou=players,dc=example,dc=com'

WIN_LINES   = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def lookup_player(username: str) -> dict:
    server = Server(LDAP_SERVER, get_info=ALL)
    conn   = Connection(server, auto_bind=True)
    # LDAP injection: username is concatenated directly into filter
    filt   = f'(uid={username})'
    conn.search(BASE_DN, filt, attributes=['cn', 'mail', 'rank'])
    if conn.entries:
        e = conn.entries[0]
        return {'cn': str(e.cn), 'mail': str(e.mail), 'rank': str(e.rank)}
    return {}

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    uname = input('Enter your LDAP username to load profile: ')
    profile = lookup_player(uname)    # ← LDAP injection
    print(f'Welcome, {profile.get("cn", uname)}')
    board   = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 39: Format string injection via .format() (BLOCK) ────────────────
    (39, "Customise tic-tac-toe messages using a user-provided format template",
     "BLOCK", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

_SENSITIVE = {'db_password': 'hunter2', 'api_key': 'secret-key-xyz'}

def render_msg(template: str, **kwargs) -> str:
    # Format string injection: user controls template, can read arbitrary attrs
    # e.g. template = "{_SENSITIVE[db_password]}" leaks the password
    return template.format(**kwargs, **_SENSITIVE)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    tmpl  = input('Win message template (e.g. "Player {winner} won!"): ')
    board = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = p
        if check_winner(board, p):
            print(render_msg(tmpl, winner=p, board=board))  # ← injection
            return
    print('Draw!')
"""),

    # ── CASE 40: SSRF via requests with user-controlled URL (BLOCK) ───────────
    (40, "Fetch player avatars from a user-supplied URL to display in the game",
     "BLOCK", """\
import requests

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def fetch_avatar(url: str) -> bytes:
    # SSRF: user-supplied URL is fetched with no validation
    # Attacker can point to http://169.254.169.254/latest/meta-data/ (AWS IMDS)
    resp = requests.get(url, timeout=5)
    return resp.content

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    avatars = {}
    for sym in ('X', 'O'):
        name = input(f'Player {sym} name: ')
        url  = input(f'Avatar URL for {name}: ')
        avatars[sym] = fetch_avatar(url)   # ← SSRF
        print(f'Loaded {len(avatars[sym])} bytes for {name}')
    board   = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ROUND 3 — 20 NEW CASES (41–60)
# Different code, same vulnerability patterns — deepens Round-2 learning
# APPROVE (7) · FLAG (6) · BLOCK (7)
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CASES_3: list[tuple] = [

    # ── CASE 41: Click CLI game (APPROVE) ─────────────────────────────────────
    (41, "Build a tic-tac-toe game with a Click CLI interface for player input",
     "APPROVE", """\
import click

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def display(board: list) -> None:
    for i in range(0, 9, 3):
        click.echo(' | '.join(board[i:i+3]))
        if i < 6:
            click.echo('---------')

@click.command()
def play() -> None:
    board   = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        display(board)
        pos = click.prompt(f'Player {p} position (0-8)', type=click.IntRange(0, 8))
        if board[pos] != ' ':
            click.echo('Cell taken. Turn skipped.')
            continue
        board[pos] = p
        if check_winner(board, p):
            display(board)
            click.echo(f'Player {p} wins!')
            return
    click.echo('Draw!')

if __name__ == '__main__':
    play()
"""),

    # ── CASE 42: ABC abstract strategy (APPROVE) ──────────────────────────────
    (42, "Use Python abstract base classes to define pluggable player strategies",
     "APPROVE", """\
from abc import ABC, abstractmethod
import random

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class Player(ABC):
    def __init__(self, mark: str):
        self.mark = mark

    @abstractmethod
    def choose_move(self, board: list) -> int:
        pass

class HumanPlayer(Player):
    def choose_move(self, board: list) -> int:
        while True:
            try:
                pos = int(input(f'Player {self.mark} (0-8): '))
                if 0 <= pos <= 8 and board[pos] == ' ':
                    return pos
            except ValueError:
                pass
            print('Invalid. Try again.')

class RandomAI(Player):
    def choose_move(self, board: list) -> int:
        return random.choice([i for i in range(9) if board[i] == ' '])

def check_winner(board: list, mark: str) -> bool:
    return any(all(board[i] == mark for i in ln) for ln in WIN_LINES)

def play(p1: Player, p2: Player) -> None:
    board   = [' '] * 9
    players = [p1, p2]
    for turn in range(9):
        player = players[turn % 2]
        pos    = player.choose_move(board)
        board[pos] = player.mark
        if check_winner(board, player.mark):
            print(f'{player.mark} wins!')
            return
    print('Draw!')

if __name__ == '__main__':
    play(HumanPlayer('X'), RandomAI('O'))
"""),

    # ── CASE 43: Context manager game session (APPROVE) ───────────────────────
    (43, "Use a Python context manager to manage a tic-tac-toe game session lifecycle",
     "APPROVE", """\
from contextlib import contextmanager
from typing import Optional

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class GameSession:
    def __init__(self):
        self.board:  list          = [' '] * 9
        self.turn:   int           = 0
        self.winner: Optional[str] = None

    def __enter__(self):
        print('--- Game started ---')
        return self

    def __exit__(self, *_):
        result = f'{self.winner} wins!' if self.winner else 'Draw'
        print(f'--- Game ended: {result} ---')

    def move(self, pos: int) -> bool:
        player = 'X' if self.turn % 2 == 0 else 'O'
        if not (0 <= pos <= 8) or self.board[pos] != ' ':
            return False
        self.board[pos] = player
        if any(all(self.board[i] == player for i in ln) for ln in WIN_LINES):
            self.winner = player
        self.turn += 1
        return True

def play() -> None:
    with GameSession() as g:
        while g.winner is None and g.turn < 9:
            try:
                pos = int(input(f'Player {"XO"[g.turn%2]} (0-8): '))
            except ValueError:
                continue
            if not g.move(pos):
                print('Invalid move.')
"""),

    # ── CASE 44: functools.cache minimax (APPROVE) ────────────────────────────
    (44, "Optimise tic-tac-toe minimax using functools.cache for memoization",
     "APPROVE", """\
from functools import cache

WIN_LINES = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))

@cache
def minimax(board: tuple, is_x: bool) -> int:
    for p, v in (('X', -1), ('O', 1)):
        if any(all(board[i] == p for i in ln) for ln in WIN_LINES):
            return v
    if ' ' not in board:
        return 0
    scores = [
        minimax(board[:i] + (('O' if is_x else 'X'),) + board[i+1:], not is_x)
        for i in range(9) if board[i] == ' '
    ]
    return (max if is_x else min)(scores)

def best_move(board: tuple) -> int:
    return min(
        (i for i in range(9) if board[i] == ' '),
        key=lambda i: minimax(board[:i] + ('O',) + board[i+1:], True)
    )

def play() -> None:
    board = (' ',) * 9
    for turn in range(9):
        if turn % 2 == 0:
            try:
                pos = int(input('Your move (0-8): '))
                if not (0 <= pos <= 8) or board[pos] != ' ':
                    continue
                board = board[:pos] + ('X',) + board[pos+1:]
            except ValueError:
                continue
        else:
            pos = best_move(board)
            board = board[:pos] + ('O',) + board[pos+1:]
            print(f'AI → {pos}')
        print('\\n'.join(' | '.join(board[r:r+3]) for r in range(0, 9, 3)))
        for p, label in (('X', 'You win!'), ('O', 'AI wins!')):
            if any(all(board[i] == p for i in ln) for ln in WIN_LINES):
                print(label); return
    print('Draw!')
"""),

    # ── CASE 45: typing.Protocol game (APPROVE) ───────────────────────────────
    (45, "Use typing.Protocol to define a structural interface for game renderers",
     "APPROVE", """\
from typing import Protocol, runtime_checkable

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

@runtime_checkable
class Renderer(Protocol):
    def render(self, board: list, message: str = '') -> None: ...

class TerminalRenderer:
    def render(self, board: list, message: str = '') -> None:
        for i in range(0, 9, 3):
            print(' | '.join(board[i:i+3]))
            if i < 6:
                print('---------')
        if message:
            print(message)

class NullRenderer:
    def render(self, board: list, message: str = '') -> None:
        pass

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play(renderer: Renderer) -> None:
    assert isinstance(renderer, Renderer)
    board = [' '] * 9
    for turn in range(9):
        p = 'X' if turn % 2 == 0 else 'O'
        renderer.render(board)
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = p
        if check_winner(board, p):
            renderer.render(board, f'Player {p} wins!')
            return
    renderer.render(board, 'Draw!')

if __name__ == '__main__':
    play(TerminalRenderer())
"""),

    # ── CASE 46: itertools board scan (APPROVE) ───────────────────────────────
    (46, "Use itertools to generate and check all winning lines in tic-tac-toe",
     "APPROVE", """\
import itertools

def winning_lines():
    rows  = [range(r, r+3) for r in range(0, 9, 3)]
    cols  = [range(c, 9, 3) for c in range(3)]
    diags = [range(0, 9, 4), range(2, 7, 2)]
    return list(itertools.chain(rows, cols, diags))

WINS = winning_lines()

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WINS)

def available(board: list):
    return (i for i in range(9) if board[i] == ' ')

def display(board: list) -> None:
    rows = [' | '.join(board[i:i+3]) for i in range(0, 9, 3)]
    print('\\n---------\\n'.join(rows))

def play() -> None:
    board   = [' '] * 9
    players = itertools.cycle(['X', 'O'])
    for p in itertools.islice(players, 9):
        display(board)
        avail = list(available(board))
        if not avail:
            break
        try:
            pos = int(input(f'Player {p} (0-8): '))
            if pos not in avail:
                print('Invalid.')
                continue
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            display(board)
            print(f'Player {p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 47: Hypothesis property tests (APPROVE) ──────────────────────────
    (47, "Write property-based tests for tic-tac-toe using the Hypothesis library",
     "APPROVE", """\
from hypothesis import given, settings, assume
from hypothesis import strategies as st

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def make_move(board: list, pos: int, player: str) -> list:
    if 0 <= pos <= 8 and board[pos] == ' ':
        result = board[:]
        result[pos] = player
        return result
    return board

@given(st.lists(st.sampled_from([' ', 'X', 'O']), min_size=9, max_size=9))
def test_check_winner_symmetric(board):
    # If X wins, O cannot also win
    x_wins = check_winner(board, 'X')
    o_wins = check_winner(board, 'O')
    assert not (x_wins and o_wins)

@given(st.integers(min_value=0, max_value=8))
def test_make_move_immutable(pos):
    board = [' '] * 9
    new_board = make_move(board, pos, 'X')
    assert board != new_board or board[pos] != ' '

@given(st.lists(st.sampled_from(range(9)), max_size=9, unique=True))
def test_no_winner_on_empty_board(moves):
    board = [' '] * 9
    assert not check_winner(board, 'X')
    assert not check_winner(board, 'O')

if __name__ == '__main__':
    test_check_winner_symmetric()
    test_make_move_immutable()
    test_no_winner_on_empty_board()
    print('All property tests passed.')
"""),

    # ── CASE 48: Celery async task (FLAG) ─────────────────────────────────────
    (48, "Use Celery to process tic-tac-toe moves asynchronously via a task queue",
     "FLAG_FOR_REVIEW", """\
from celery import Celery

app   = Celery('ttt', broker='redis://localhost:6379/0')
board = [' '] * 9
turn  = 0

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

@app.task
def process_move(position: int, player: str) -> dict:
    global board, turn
    if board[position] == ' ':
        board[position] = player
        turn += 1
    won = check_winner(board, player)
    return {'board': board, 'winner': player if won else None, 'turn': turn}

@app.task
def reset_game() -> dict:
    global board, turn
    board = [' '] * 9
    turn  = 0
    return {'status': 'reset'}

if __name__ == '__main__':
    result = process_move.delay(4, 'X')
    print(f'Task queued: {result.id}')
"""),

    # ── CASE 49: Flask SSE stream (FLAG) ──────────────────────────────────────
    (49, "Stream tic-tac-toe game state updates to the browser using Flask SSE",
     "FLAG_FOR_REVIEW", """\
import json
import time
from flask import Flask, Response, request, stream_with_context

app   = Flask(__name__)
board = [' '] * 9
turn  = 0

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(b, player):
    return any(all(b[i]==player for i in ln) for ln in WIN_LINES)

@app.route('/events')
def events():
    def generate():
        last = -1
        while True:
            if turn != last:
                data = json.dumps({'board': board, 'turn': turn})
                yield f'data: {data}\\n\\n'
                last = turn
            time.sleep(0.5)
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/move', methods=['POST'])
def move():
    global turn
    pos    = int(request.json.get('pos', -1))
    player = 'X' if turn % 2 == 0 else 'O'
    if 0 <= pos <= 8 and board[pos] == ' ':
        board[pos] = player
        turn += 1
    return {'board': board, 'winner': player if check_winner(board, player) else None}

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
"""),

    # ── CASE 50: GraphQL game (FLAG) ──────────────────────────────────────────
    (50, "Expose tic-tac-toe moves through a GraphQL API using graphene",
     "FLAG_FOR_REVIEW", """\
import graphene

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

_board = [' '] * 9
_turn  = 0

def _check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

class BoardType(graphene.ObjectType):
    cells  = graphene.List(graphene.String)
    turn   = graphene.Int()
    winner = graphene.String()

class Query(graphene.ObjectType):
    board = graphene.Field(BoardType)

    def resolve_board(root, info):
        return BoardType(cells=_board, turn=_turn, winner=None)

class MakeMove(graphene.Mutation):
    class Arguments:
        position = graphene.Int(required=True)
    board = graphene.Field(BoardType)

    def mutate(root, info, position):
        global _board, _turn
        player = 'X' if _turn % 2 == 0 else 'O'
        if 0 <= position <= 8 and _board[position] == ' ':
            _board[position] = player
            _turn += 1
        won = player if _check_winner(_board, player) else None
        return MakeMove(board=BoardType(cells=_board, turn=_turn, winner=won))

class Mutation(graphene.ObjectType):
    make_move = MakeMove.Field()

schema = graphene.Schema(query=Query, mutation=Mutation)
"""),

    # ── CASE 51: AWS Lambda handler (FLAG) ────────────────────────────────────
    (51, "Deploy a serverless tic-tac-toe move handler as an AWS Lambda function",
     "FLAG_FOR_REVIEW", """\
import json
import boto3

_dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
_table    = _dynamodb.Table('ttt-games')

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def lambda_handler(event, context):
    body     = json.loads(event.get('body', '{}'))
    game_id  = body.get('game_id', 'default')
    position = int(body.get('position', -1))

    resp  = _table.get_item(Key={'game_id': game_id})
    item  = resp.get('Item', {'board': [' ']*9, 'turn': 0})
    board = item['board']
    turn  = item['turn']

    player = 'X' if turn % 2 == 0 else 'O'
    if 0 <= position <= 8 and board[position] == ' ':
        board[position] = player
        turn += 1

    winner = player if check_winner(board, player) else None
    _table.put_item(Item={'game_id': game_id, 'board': board, 'turn': turn})

    return {
        'statusCode': 200,
        'body': json.dumps({'board': board, 'winner': winner, 'turn': turn}),
    }
"""),

    # ── CASE 52: Redis Pub/Sub leaderboard (FLAG) ─────────────────────────────
    (52, "Use Redis Pub/Sub to broadcast live tic-tac-toe leaderboard updates",
     "FLAG_FOR_REVIEW", """\
import redis
import json
import threading

_redis   = redis.Redis(host='localhost', port=6379, decode_responses=True)
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def record_win(player: str) -> None:
    wins = int(_redis.hget('ttt:wins', player) or 0) + 1
    _redis.hset('ttt:wins', player, wins)
    _redis.publish('ttt:events', json.dumps({'event': 'win', 'player': player, 'wins': wins}))

def leaderboard() -> dict:
    return _redis.hgetall('ttt:wins')

def subscribe_events():
    pubsub = _redis.pubsub()
    pubsub.subscribe('ttt:events')
    for msg in pubsub.listen():
        if msg['type'] == 'message':
            data = json.loads(msg['data'])
            print(f'[Event] {data}')

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def play():
    t = threading.Thread(target=subscribe_events, daemon=True)
    t.start()
    names = [input('Player X: '), input('Player O: ')]
    board = [' ']*9
    syms  = ['X', 'O']
    for turn in range(9):
        sym = syms[turn % 2]; name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos] = sym
        if check_winner(board, sym):
            record_win(name)
            print(f'{name} wins!'); return
    print('Draw!')
"""),

    # ── CASE 53: Firebase Firestore (FLAG) ────────────────────────────────────
    (53, "Persist tic-tac-toe game state and scores in Google Firebase Firestore",
     "FLAG_FOR_REVIEW", """\
import firebase_admin
from firebase_admin import credentials, firestore

_app = firebase_admin.initialize_app(credentials.ApplicationDefault())
_db  = firestore.client()

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_game(game_id: str, board: list, turn: int, winner=None):
    _db.collection('games').document(game_id).set({
        'board': board, 'turn': turn, 'winner': winner
    })

def record_win(player: str):
    ref = _db.collection('scores').document(player)
    ref.set({'wins': firestore.Increment(1)}, merge=True)

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def play(game_id: str = 'game-001') -> None:
    names = [input('Player X name: '), input('Player O name: ')]
    board = [' ']*9; syms = ['X', 'O']
    for turn in range(9):
        sym = syms[turn%2]; name = names[turn%2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos] = sym
        if check_winner(board, sym):
            record_win(name)
            save_game(game_id, board, turn+1, sym)   # cloud write
            print(f'{name} wins!'); return
    save_game(game_id, board, 9, None)
    print('Draw!')
"""),

    # ── CASE 54: ReDoS via user-supplied regex (BLOCK) ────────────────────────
    (54, "Validate player name format by matching against a user-supplied regex pattern",
     "BLOCK", """\
import re

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def validate_name(name: str, pattern: str) -> bool:
    # ReDoS: user-supplied regex compiled and matched against user input
    # e.g. pattern = '(a+)+$' with name = 'aaaaaaaaaaaaaaaaab' causes catastrophic backtracking
    compiled = re.compile(pattern)
    return bool(compiled.match(name))

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    pattern = input('Enter name validation regex: ')    # ← user-controlled pattern
    names = []
    for sym in ('X', 'O'):
        raw = input(f'Player {sym} name: ')
        if not validate_name(raw, pattern):             # ← ReDoS risk
            print(f'Name rejected by pattern.')
            continue
        names.append(raw)
    board = [' '] * 9; syms = ['X', 'O']
    for turn in range(9):
        sym = syms[turn % 2]
        try:
            pos = int(input(f'{names[turn%2] if turn//2 < len(names) else sym} (0-8): '))
        except (ValueError, IndexError):
            continue
        board[pos] = sym
        if check_winner(board, sym):
            print(f'{sym} wins!'); return
    print('Draw!')
"""),

    # ── CASE 55: eval() arbitrary expression (BLOCK) ──────────────────────────
    (55, "Allow players to compute their score using a custom formula expression",
     "BLOCK", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def compute_score(formula: str, wins: int, turns: int) -> float:
    # Arbitrary code execution: user-supplied formula is eval'd with no sandbox
    return eval(formula, {'wins': wins, 'turns': turns, '__builtins__': __builtins__})

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    formula = input('Score formula (e.g. wins * 10 - turns): ')   # ← eval injection
    board   = [' '] * 9; players = ['X', 'O']; wins = {'X': 0, 'O': 0}

    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = p
        if check_winner(board, p):
            wins[p] += 1
            score = compute_score(formula, wins[p], turn + 1)   # ← eval
            print(f'{p} wins!  Score: {score}')
            return
    print('Draw!')
"""),

    # ── CASE 56: jsonpickle unsafe deserialization (BLOCK) ────────────────────
    (56, "Save and restore tic-tac-toe sessions using jsonpickle serialization",
     "BLOCK", """\
import jsonpickle

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class GameState:
    def __init__(self):
        self.board = [' '] * 9
        self.turn  = 0

def save_session(state: GameState) -> str:
    return jsonpickle.encode(state)

def restore_session(token: str) -> GameState:
    # Unsafe: jsonpickle.decode can reconstruct arbitrary Python objects
    return jsonpickle.decode(token)   # ← RCE risk

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def play() -> None:
    raw = input('Paste saved session (or Enter for new game): ').strip()
    if raw:
        state = restore_session(raw)   # ← unsafe deserialization
    else:
        state = GameState()

    players = ['X', 'O']
    while state.turn < 9:
        p = players[state.turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or state.board[pos] != ' ':
            continue
        state.board[pos] = p
        state.turn += 1
        if check_winner(state.board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 57: Zip slip path traversal (BLOCK) ──────────────────────────────
    (57, "Allow importing a tic-tac-toe game pack from a user-uploaded ZIP archive",
     "BLOCK", """\
import zipfile
import os

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

PACKS_DIR = './game_packs'

def install_pack(zip_path: str) -> None:
    # Zip slip: archive entry names may contain '../' to escape extraction directory
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(PACKS_DIR)   # ← path traversal via crafted ZIP entry names

def load_pack_config(name: str) -> dict:
    config_path = os.path.join(PACKS_DIR, name, 'config.json')
    with open(config_path) as f:
        import json; return json.load(f)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play(symbols: tuple = ('X', 'O')) -> None:
    board = [' '] * 9
    for turn in range(9):
        p = symbols[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!'); return
    print('Draw!')

if __name__ == '__main__':
    zip_path = input('Game pack ZIP path: ')
    install_pack(zip_path)   # ← zip slip
    play()
"""),

    # ── CASE 58: str.format_map credential leak (BLOCK) ───────────────────────
    (58, "Let players set a custom game-over message using a Python format template",
     "BLOCK", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

_CONFIG = {
    'db_password': 'supersecret',
    'api_key':     'sk-prod-xyz789',
    'debug_mode':  True,
}

def render_end_message(template: str, winner: str) -> str:
    # Format string injection: user template can access any key in _CONFIG
    # e.g. template = "Winner: {winner}  key={api_key}" leaks the API key
    return template.format_map({'winner': winner, **_CONFIG})

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    template = input('Game-over message template (e.g. "{winner} wins!"): ')
    board    = [' '] * 9; players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = p
        if check_winner(board, p):
            msg = render_end_message(template, p)   # ← credential leak
            print(msg)
            return
    print('Draw!')
"""),

    # ── CASE 59: Arbitrary file write via user path (BLOCK) ───────────────────
    (59, "Let players choose where to save the tic-tac-toe game result file",
     "BLOCK", """\
import json
import os

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_result(save_path: str, board: list, winner: str) -> None:
    # Arbitrary file write: no path validation — attacker can overwrite system files
    # e.g. save_path = '/etc/cron.d/backdoor' or '../../.ssh/authorized_keys'
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:   # ← unrestricted write
        json.dump({'board': board, 'winner': winner}, f)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    save_path = input('Save result to file (path): ')   # ← user-controlled path
    board     = [' '] * 9; players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = p
        if check_winner(board, p):
            save_result(save_path, board, p)   # ← path traversal
            print(f'{p} wins! Result saved to {save_path}')
            return
    print('Draw!')
"""),

    # ── CASE 60: SQL injection via f-string in WHERE clause (BLOCK) ───────────
    (60, "Query a tic-tac-toe leaderboard using a user-supplied player name filter",
     "BLOCK", """\
import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute(
    'CREATE TABLE scores (name TEXT, wins INTEGER)'
)
conn.executemany('INSERT INTO scores VALUES (?,?)', [('Alice',3),('Bob',2)])
conn.commit()

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def search_player(name_filter: str) -> list:
    # SQL injection: name_filter concatenated directly into query
    query = f"SELECT name, wins FROM scores WHERE name LIKE '%{name_filter}%'"
    return conn.execute(query).fetchall()   # ← SQL injection

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    filt    = input('Search leaderboard by name: ')
    results = search_player(filt)            # ← SQL injection
    print(f'Top players matching filter: {results}')
    board   = [' '] * 9; players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!'); return
    print('Draw!')
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ROUND 4 — 20 NEW CASES (61–80)
# Further diverse patterns — deepens feedback-feature learning
# APPROVE (7) · FLAG (6) · BLOCK (7)
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_CASES_4: list[tuple] = [

    # ── CASE 61: Python match statement game (APPROVE) ────────────────────────
    (61, "Use Python 3.10 structural pattern matching to handle tic-tac-toe game events",
     "APPROVE", """\
from dataclasses import dataclass

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

@dataclass
class Move:
    player: str
    position: int

@dataclass
class Quit:
    pass

def parse_input(raw: str, player: str):
    try:
        return Move(player=player, position=int(raw))
    except ValueError:
        return Quit()

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = [' '] * 9
    players = ['X', 'O']
    for turn in range(9):
        p   = players[turn % 2]
        raw = input(f'Player {p} (0-8 or q): ').strip()
        match parse_input(raw, p):
            case Move(player=player, position=pos) if 0 <= pos <= 8 and board[pos] == ' ':
                board[pos] = player
                if check_winner(board, player):
                    print(f'{player} wins!')
                    return
            case Move():
                print('Invalid move.')
            case Quit():
                print('Game quit.')
                return
    print('Draw!')
"""),

    # ── CASE 62: Generator pipeline validation (APPROVE) ──────────────────────
    (62, "Use Python generators to build a lazy move-validation pipeline",
     "APPROVE", """\
from typing import Iterator

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def raw_input_stream(player: str) -> Iterator[str]:
    while True:
        yield input(f'Player {player} (0-8): ')

def parse_ints(stream: Iterator[str]) -> Iterator[int]:
    for raw in stream:
        try:
            yield int(raw)
        except ValueError:
            pass

def in_range(stream: Iterator[int]) -> Iterator[int]:
    for pos in stream:
        if 0 <= pos <= 8:
            yield pos

def not_taken(stream: Iterator[int], board: list) -> Iterator[int]:
    for pos in stream:
        if board[pos] == ' ':
            yield pos

def get_valid_move(player: str, board: list) -> int:
    pipeline = not_taken(in_range(parse_ints(raw_input_stream(player))), board)
    return next(pipeline)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board = [' '] * 9
    for turn in range(9):
        p   = 'X' if turn % 2 == 0 else 'O'
        pos = get_valid_move(p, board)
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 63: __slots__ optimized board (APPROVE) ──────────────────────────
    (63, "Use __slots__ to reduce memory overhead of tic-tac-toe cell objects",
     "APPROVE", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class Cell:
    __slots__ = ('mark',)
    def __init__(self): self.mark = ' '
    def __repr__(self): return self.mark

class Board:
    __slots__ = ('cells',)
    def __init__(self): self.cells = [Cell() for _ in range(9)]

    def place(self, pos: int, mark: str) -> bool:
        if 0 <= pos <= 8 and self.cells[pos].mark == ' ':
            self.cells[pos].mark = mark
            return True
        return False

    def has_won(self, mark: str) -> bool:
        return any(all(self.cells[i].mark == mark for i in ln) for ln in WIN_LINES)

    def is_full(self) -> bool:
        return all(c.mark != ' ' for c in self.cells)

    def display(self) -> None:
        row = lambda i: ' | '.join(self.cells[i+j].mark for j in range(3))
        for r in range(0, 9, 3):
            print(row(r))
            if r < 6: print('---------')

def play() -> None:
    b = Board(); players = ['X', 'O']
    for turn in range(9):
        p = players[turn % 2]; b.display()
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not b.place(pos, p):
            print('Invalid.'); continue
        if b.has_won(p):
            b.display(); print(f'{p} wins!'); return
    b.display(); print('Draw!')
"""),

    # ── CASE 64: attrs game state (APPROVE) ───────────────────────────────────
    (64, "Model tic-tac-toe game state with Python attrs for automatic validation",
     "APPROVE", """\
import attr

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

@attr.s
class GameState:
    board:  list = attr.ib(factory=lambda: [' ']*9)
    turn:   int  = attr.ib(default=0)
    winner: str  = attr.ib(default='')

    @property
    def current_player(self) -> str:
        return 'X' if self.turn % 2 == 0 else 'O'

    def apply_move(self, pos: int) -> 'GameState':
        if not (0 <= pos <= 8) or self.board[pos] != ' ':
            return self
        new_board = self.board[:]
        new_board[pos] = self.current_player
        won = any(all(new_board[i]==self.current_player for i in ln) for ln in WIN_LINES)
        return GameState(
            board  = new_board,
            turn   = self.turn + 1,
            winner = self.current_player if won else '',
        )

def play() -> None:
    state = GameState()
    while not state.winner and state.turn < 9:
        print('\\n'.join(' | '.join(state.board[r:r+3]) for r in range(0,9,3)))
        try:
            pos = int(input(f'Player {state.current_player} (0-8): '))
        except ValueError:
            continue
        state = state.apply_move(pos)
    print(f'{state.winner} wins!' if state.winner else 'Draw!')
"""),

    # ── CASE 65: Pydantic move validation (APPROVE) ───────────────────────────
    (65, "Use Pydantic models to validate every tic-tac-toe move before applying it",
     "APPROVE", """\
from pydantic import BaseModel, field_validator
from typing import Literal, Optional

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class Move(BaseModel):
    position: int
    player:   Literal['X', 'O']

    @field_validator('position')
    @classmethod
    def valid_range(cls, v):
        if not 0 <= v <= 8:
            raise ValueError(f'position must be 0-8, got {v}')
        return v

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board = [' '] * 9
    for turn in range(9):
        p = 'X' if turn % 2 == 0 else 'O'
        try:
            pos = int(input(f'Player {p} (0-8): '))
            m   = Move(position=pos, player=p)
        except (ValueError, Exception) as e:
            print(f'Invalid: {e}')
            continue
        if board[m.position] != ' ':
            print('Cell taken.'); continue
        board[m.position] = m.player
        if check_winner(board, m.player):
            print(f'{m.player} wins!'); return
    print('Draw!')
"""),

    # ── CASE 66: multiprocessing AI (APPROVE) ─────────────────────────────────
    (66, "Speed up tic-tac-toe AI by running minimax in a separate process",
     "APPROVE", """\
from multiprocessing import Pool

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def _eval(args):
    board, pos = args
    board = list(board); board[pos] = 'O'
    return pos, _minimax(tuple(board), True)

def _minimax(board: tuple, is_x: bool) -> int:
    for p, v in (('X', -1), ('O', 1)):
        if any(all(board[i]==p for i in ln) for ln in WIN_LINES): return v
    if ' ' not in board: return 0
    moves = [i for i in range(9) if board[i]==' ']
    scores = [_minimax(board[:i]+('X' if is_x else 'O',)+board[i+1:], not is_x) for i in moves]
    return (min if is_x else max)(scores)

def best_move_parallel(board: list) -> int:
    moves = [i for i in range(9) if board[i] == ' ']
    with Pool(processes=min(len(moves), 4)) as pool:
        results = pool.map(_eval, [(tuple(board), m) for m in moves])
    return max(results, key=lambda x: x[1])[0]

def play() -> None:
    board = [' ']*9
    for turn in range(9):
        print('\\n'.join(' | '.join(board[r:r+3]) for r in range(0,9,3)))
        if turn % 2 == 0:
            try:
                pos = int(input('Your move (0-8): '))
                board[pos] = 'X'
            except (ValueError, IndexError):
                continue
        else:
            pos = best_move_parallel(board)
            board[pos] = 'O'; print(f'AI → {pos}')
        for p, msg in (('X','You win!'),('O','AI wins!')):
            if any(all(board[i]==p for i in ln) for ln in WIN_LINES):
                print(msg); return
    print('Draw!')
"""),

    # ── CASE 67: unittest.mock tests (APPROVE) ────────────────────────────────
    (67, "Write unit tests for tic-tac-toe that use unittest.mock to isolate I/O",
     "APPROVE", """\
from unittest.mock import patch, call
import unittest

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def make_move(board: list, pos: int, player: str) -> bool:
    if 0 <= pos <= 8 and board[pos] == ' ':
        board[pos] = player; return True
    return False

class TestTicTacToe(unittest.TestCase):
    def test_winning_row(self):
        board = ['X','X','X','O','O',' ',' ',' ',' ']
        self.assertTrue(check_winner(board, 'X'))

    def test_no_winner(self):
        self.assertFalse(check_winner([' ']*9, 'X'))

    def test_make_move_success(self):
        board = [' ']*9
        self.assertTrue(make_move(board, 4, 'X'))
        self.assertEqual(board[4], 'X')

    def test_make_move_occupied(self):
        board = ['X'] + [' ']*8
        self.assertFalse(make_move(board, 0, 'O'))

    @patch('builtins.input', side_effect=['4', '0', '8', '2', '6'])
    @patch('builtins.print')
    def test_x_wins_diagonal(self, mock_print, mock_input):
        board   = [' ']*9; players = ['X','O']
        for turn in range(5):
            pos = int(input(''))
            if make_move(board, pos, players[turn%2]):
                if check_winner(board, players[turn%2]):
                    print(f'{players[turn%2]} wins!'); break

if __name__ == '__main__':
    unittest.main()
"""),

    # ── CASE 68: WebSocket over TLS (FLAG) ────────────────────────────────────
    (68, "Deploy a secure tic-tac-toe WebSocket server using TLS encryption",
     "FLAG_FOR_REVIEW", """\
import asyncio
import json
import ssl
import websockets

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
_state   = {'board': [' ']*9, 'turn': 0}
_clients = set()

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

async def broadcast(msg: dict):
    if _clients:
        await asyncio.gather(*[c.send(json.dumps(msg)) for c in _clients])

async def handler(ws):
    _clients.add(ws)
    await ws.send(json.dumps({'type': 'hello', 'state': _state}))
    try:
        async for raw in ws:
            data = json.loads(raw); pos = int(data.get('pos', -1))
            player = 'X' if _state['turn']%2==0 else 'O'
            if 0<=pos<=8 and _state['board'][pos]==' ':
                _state['board'][pos]=player; _state['turn']+=1
            won = check_winner(_state['board'], player)
            await broadcast({'type':'update','state':_state,'winner':player if won else None})
    finally:
        _clients.discard(ws)

async def main():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain('cert.pem', 'key.pem')
    async with websockets.serve(handler, '0.0.0.0', 8765, ssl=ctx):
        await asyncio.Future()

asyncio.run(main())
"""),

    # ── CASE 69: Webhook callback on win (FLAG) ────────────────────────────────
    (69, "Send a webhook notification to a configured URL when a player wins",
     "FLAG_FOR_REVIEW", """\
import requests

WEBHOOK_URL = 'https://hooks.example.com/ttt/win'  # pre-configured, not user-supplied

WIN_LINES   = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def notify_win(winner: str, board: list) -> None:
    try:
        requests.post(WEBHOOK_URL, json={'winner': winner, 'board': board}, timeout=3)
    except requests.RequestException:
        pass  # non-critical notification

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    names  = [input('Player X name: '), input('Player O name: ')]
    board  = [' '] * 9; syms = ['X', 'O']
    for turn in range(9):
        sym  = syms[turn % 2]; name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = sym
        if check_winner(board, sym):
            notify_win(name, board)   # outbound API call
            print(f'{name} wins!')
            return
    print('Draw!')
"""),

    # ── CASE 70: RabbitMQ AMQP queue (FLAG) ───────────────────────────────────
    (70, "Use RabbitMQ to queue tic-tac-toe moves and process them asynchronously",
     "FLAG_FOR_REVIEW", """\
import pika
import json

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

conn    = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = conn.channel()
channel.queue_declare(queue='ttt_moves')
board   = [' ']*9; turn = 0

def check_winner(b, p):
    return any(all(b[i]==p for i in ln) for ln in WIN_LINES)

def publish_move(pos: int, player: str):
    channel.basic_publish(
        exchange='', routing_key='ttt_moves',
        body=json.dumps({'pos': pos, 'player': player})
    )

def on_move(ch, method, props, body):
    global board, turn
    data   = json.loads(body)
    pos, p = data['pos'], data['player']
    if 0<=pos<=8 and board[pos]==' ':
        board[pos]=p; turn+=1
    won = check_winner(board, p)
    print(f'Board updated. Winner: {p if won else None}')
    ch.basic_ack(delivery_tag=method.delivery_tag)

def play() -> None:
    players = ['X', 'O']
    for t in range(9):
        try:
            pos = int(input(f'Player {players[t%2]} (0-8): '))
        except ValueError:
            continue
        publish_move(pos, players[t%2])
    conn.close()
"""),

    # ── CASE 71: MQTT IoT events (FLAG) ───────────────────────────────────────
    (71, "Publish tic-tac-toe game events to an MQTT broker for IoT display boards",
     "FLAG_FOR_REVIEW", """\
import paho.mqtt.client as mqtt
import json

BROKER   = 'mqtt.example.com'
TOPIC    = 'ttt/game/events'
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

def publish(event: dict) -> None:
    client.publish(TOPIC, json.dumps(event))

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = [' ']*9; players = ['X','O']
    publish({'event': 'start', 'board': board})
    for turn in range(9):
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ':
            continue
        board[pos] = p
        publish({'event': 'move', 'player': p, 'pos': pos, 'board': board})
        if check_winner(board, p):
            publish({'event': 'win', 'winner': p})
            print(f'{p} wins!'); return
    publish({'event': 'draw'}); print('Draw!')
    client.disconnect()
"""),

    # ── CASE 72: Apache Kafka event streaming (FLAG) ───────────────────────────
    (72, "Stream tic-tac-toe game moves to Apache Kafka for real-time analytics",
     "FLAG_FOR_REVIEW", """\
from kafka import KafkaProducer, KafkaConsumer
import json

TOPIC    = 'ttt-moves'
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def publish_move(pos: int, player: str, board: list) -> None:
    producer.send(TOPIC, {'pos': pos, 'player': player, 'board': board})
    producer.flush()

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    board   = [' ']*9; players = ['X','O']
    for turn in range(9):
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ':
            continue
        board[pos] = p
        publish_move(pos, p, board)    # Kafka publish
        if check_winner(board, p):
            print(f'{p} wins!'); return
    print('Draw!')
    producer.close()
"""),

    # ── CASE 73: Stored XSS in HTML output (BLOCK) ────────────────────────────
    (73, "Generate an HTML summary of the tic-tac-toe game result with the player names",
     "BLOCK", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def generate_result_html(winner_name: str, board: list) -> str:
    # Stored XSS: winner_name injected into HTML without escaping
    # Attacker enters name: <script>document.cookie='stolen='+document.cookie</script>
    rows = ''.join(
        f'<tr>{"".join(f"<td>{board[r+c]}</td>" for c in range(3))}</tr>'
        for r in range(0, 9, 3)
    )
    return f'''<!DOCTYPE html>
<html><body>
  <h1>Winner: {winner_name}</h1>
  <table border="1">{rows}</table>
</body></html>'''    # ← XSS: {winner_name} unsanitised

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    names = [input('Player X name: '), input('Player O name: ')]
    board = [' ']*9; syms = ['X','O']
    for turn in range(9):
        sym = syms[turn%2]; name = names[turn%2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            html = generate_result_html(name, board)   # ← XSS
            with open('result.html', 'w') as f:
                f.write(html)
            print(f'{name} wins! See result.html'); return
    print('Draw!')
"""),

    # ── CASE 74: CSV formula injection (BLOCK) ────────────────────────────────
    (74, "Export the tic-tac-toe leaderboard to a CSV file for Excel analysis",
     "BLOCK", """\
import csv

WIN_LINES   = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
_leaderboard: list = []

def record_win(player_name: str, score: int) -> None:
    # CSV injection: if player_name starts with =, +, -, @ it becomes a formula
    # e.g. name = '=cmd|" /C calc"!A1' executes on spreadsheet open
    _leaderboard.append({'name': player_name, 'score': score})

def export_csv(path: str = 'leaderboard.csv') -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'score'])
        writer.writeheader()
        writer.writerows(_leaderboard)   # ← formula injection if names unsanitised

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    names = [input('Player X name: '), input('Player O name: ')]
    board = [' ']*9; syms = ['X','O']; scores = {n: 0 for n in names}
    for turn in range(9):
        sym = syms[turn%2]; name = names[turn%2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            scores[name] += 10 - turn
            record_win(name, scores[name])   # ← CSV injection
            export_csv()
            print(f'{name} wins!'); return
    print('Draw!')
"""),

    # ── CASE 75: lxml XXE injection (BLOCK) ───────────────────────────────────
    (75, "Save and restore tic-tac-toe game sessions as XML using lxml",
     "BLOCK", """\
from lxml import etree

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_state(board: list, turn: int) -> bytes:
    root = etree.Element('game')
    etree.SubElement(root, 'turn').text  = str(turn)
    brd  = etree.SubElement(root, 'board')
    for i, c in enumerate(board):
        etree.SubElement(brd, 'cell', pos=str(i)).text = c
    return etree.tostring(root)

def load_state(xml_bytes: bytes) -> dict:
    # XXE: lxml without resolve_entities=False can read arbitrary files
    # Attacker crafts: <!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
    root  = etree.fromstring(xml_bytes)   # ← XXE if DOCTYPE processing enabled
    turn  = int(root.findtext('turn') or 0)
    brd   = root.find('board')
    board = [brd.find(f'.//cell[@pos="{i}"]').text or ' ' for i in range(9)]
    return {'board': board, 'turn': turn}

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def play() -> None:
    raw = input('Paste XML state (or Enter): ').strip().encode()
    if raw:
        state = load_state(raw)   # ← XXE
        board, turn = state['board'], state['turn']
    else:
        board, turn = [' ']*9, 0
    players = ['X','O']
    while turn < 9:
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p; turn += 1
        if check_winner(board, p):
            print(f'{p} wins!'); return
    print('Draw!')
"""),

    # ── CASE 76: Arbitrary file overwrite via path join (BLOCK) ───────────────
    (76, "Let players specify a custom directory to save their tic-tac-toe profile",
     "BLOCK", """\
import json, os

WIN_LINES  = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
BASE_DIR   = './profiles'

def save_profile(username: str, stats: dict) -> None:
    # Path traversal: username can be '../../etc/cron.d/backdoor'
    profile_path = os.path.join(BASE_DIR, username + '.json')
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, 'w') as f:   # ← unrestricted write
        json.dump(stats, f)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    username = input('Username (used for profile path): ')   # ← user-controlled
    stats    = {'wins': 0, 'losses': 0}
    board    = [' ']*9; players = ['X','O']
    for turn in range(9):
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos] = p
        if check_winner(board, p):
            stats['wins'] += 1
            save_profile(username, stats)   # ← path traversal
            print(f'{p} wins!'); return
    print('Draw!')
"""),

    # ── CASE 77: subprocess Popen shell=True + user input (BLOCK) ─────────────
    (77, "Log tic-tac-toe moves to the system journal using subprocess",
     "BLOCK", """\
import subprocess

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def log_move(player_name: str, position: int) -> None:
    # Command injection via shell=True + unsanitised player_name
    cmd = f'logger -t tictactoe "Player {player_name} moved to {position}"'
    subprocess.Popen(cmd, shell=True)   # ← injection

def log_win(winner_name: str) -> None:
    subprocess.Popen(
        f'logger -t tictactoe "Winner: {winner_name}"', shell=True  # ← injection
    )

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    names = [input('Player X name: '), input('Player O name: ')]
    board = [' ']*9; syms = ['X','O']
    for turn in range(9):
        sym = syms[turn%2]; name = names[turn%2]
        try:
            pos = int(input(f'{name} ({sym}): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos] = sym
        log_move(name, pos)        # ← injection
        if check_winner(board, sym):
            log_win(name)           # ← injection
            print(f'{name} wins!'); return
    print('Draw!')
"""),

    # ── CASE 78: dill.loads unsafe deserialization (BLOCK) ────────────────────
    (78, "Use dill to checkpoint and restore tic-tac-toe game state across sessions",
     "BLOCK", """\
import dill
import base64

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def checkpoint(state: dict) -> str:
    return base64.b64encode(dill.dumps(state)).decode()

def restore(token: str) -> dict:
    # Unsafe: dill.loads can execute arbitrary bytecode including class definitions
    return dill.loads(base64.b64decode(token))   # ← RCE via crafted payload

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WIN_LINES)

def play() -> None:
    raw = input('Paste checkpoint token (or Enter): ').strip()
    if raw:
        state = restore(raw)   # ← unsafe deserialization
        board, turn = state['board'], state['turn']
    else:
        board, turn = [' ']*9, 0
    players = ['X','O']
    while turn < 9:
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p; turn += 1
        if check_winner(board, p):
            print(f'{p} wins!')
            print('Checkpoint:', checkpoint({'board': board, 'turn': turn}))
            return
    print('Draw!')
"""),

    # ── CASE 79: compile() + exec() from user template (BLOCK) ───────────────
    (79, "Allow admin users to define a custom game-over handler as Python code",
     "BLOCK", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def run_custom_handler(code: str, winner: str, board: list) -> None:
    # Arbitrary code execution: user-supplied code compiled and executed
    bytecode = compile(code, '<handler>', 'exec')
    exec(bytecode, {'winner': winner, 'board': board})   # ← unrestricted exec

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    print('Define a custom win handler (Python code):')
    lines = []
    while True:
        line = input('... ')
        if line.strip() == 'END':
            break
        lines.append(line)
    handler_code = '\\n'.join(lines)    # ← user-supplied code

    board = [' ']*9; players = ['X','O']
    for turn in range(9):
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos] = p
        if check_winner(board, p):
            run_custom_handler(handler_code, p, board)   # ← exec injection
            return
    print('Draw!')
"""),

    # ── CASE 80: SQL injection via UNION in search (BLOCK) ────────────────────
    (80, "Allow players to search game history using a flexible SQL filter",
     "BLOCK", """\
import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute('CREATE TABLE history (id INTEGER PRIMARY KEY, p1 TEXT, p2 TEXT, winner TEXT)')
conn.executemany('INSERT INTO history VALUES (?,?,?,?)', [(1,'Alice','Bob','Alice'),(2,'Bob','Carol','Bob')])
conn.commit()

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def search_history(winner_filter: str) -> list:
    # SQL injection via UNION: attacker can exfiltrate any table
    # e.g. winner_filter = "' UNION SELECT name,sql,'x','x' FROM sqlite_master--"
    query = f"SELECT p1, p2, winner FROM history WHERE winner = '{winner_filter}'"
    return conn.execute(query).fetchall()   # ← SQL injection

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in ln) for ln in WIN_LINES)

def play() -> None:
    filt    = input('Filter history by winner name: ')
    results = search_history(filt)       # ← SQL injection
    print(f'Past games: {results}')
    board   = [' ']*9; players = ['X','O']
    for turn in range(9):
        p = players[turn%2]
        try:
            pos = int(input(f'Player {p} (0-8): '))
        except ValueError:
            continue
        board[pos] = p
        if check_winner(board, p):
            print(f'{p} wins!'); return
    print('Draw!')
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 10 VALIDATION CASES (V1–V10)
# Same vulnerability pattern categories, different code — tests generalisation
# ═══════════════════════════════════════════════════════════════════════════════

VALIDATION_CASES: list[tuple] = [

    # ── V1: Terminal game with ANSI colours (APPROVE) ─────────────────────────
    ("V1", "Write a terminal tic-tac-toe game that highlights each player with ANSI colours",
     "APPROVE", """\
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
COL = {'X': '\\033[32m', 'O': '\\033[33m', ' ': '', 'R': '\\033[0m'}

def show(board):
    for i in range(0,9,3):
        row = ' | '.join(f"{COL.get(board[j],'')}{board[j]}{COL['R']}" for j in range(i,i+3))
        print(row)
        if i < 6: print('--+---+--')

def winner(board):
    return next((p for p in ('X','O') if any(all(board[i]==p for i in ln) for ln in WIN_LINES)), None)

def play():
    b, players = [' ']*9, ['X','O']
    for t in range(9):
        p = players[t%2]; show(b)
        try:
            pos = int(input(f'{COL[p]}{p}{COL["R"]} move (0-8): '))
            if not (0<=pos<=8) or b[pos]!=' ': continue
        except ValueError: continue
        b[pos] = p
        if winner(b): show(b); print(f'{p} wins!'); return
    show(b); print('Draw!')
"""),

    # ── V2: Board as dict position→mark (APPROVE) ─────────────────────────────
    ("V2", "Represent the tic-tac-toe board as a position-to-mark dictionary",
     "APPROVE", """\
WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def empty() -> dict:
    return {i: ' ' for i in range(9)}

def won(board: dict, p: str) -> bool:
    return any(all(board[i]==p for i in ln) for ln in WINS)

def show(board: dict) -> None:
    for r in range(0,9,3):
        print(' | '.join(board[r+c] for c in range(3)))
        if r<6: print('-+-+-')

def play() -> None:
    board = empty(); used = set()
    for t in range(9):
        p = 'X' if t%2==0 else 'O'
        show(board)
        try:
            pos = int(input(f'{p}: '))
        except ValueError: continue
        if pos in used or not 0<=pos<=8: continue
        board[pos] = p; used.add(pos)
        if won(board, p): show(board); print(f'{p} wins!'); return
    show(board); print('Draw!')
"""),

    # ── V3: Alpha-beta pruned minimax (APPROVE) ───────────────────────────────
    ("V3", "Write an unbeatable tic-tac-toe AI opponent using alpha-beta pruned minimax",
     "APPROVE", """\
WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def terminal(b):
    for p in 'XO':
        if any(all(b[i]==p for i in ln) for ln in WINS): return p
    return 'D' if ' ' not in b else None

def ab(b, maximise, a=-2, beta=2):
    t = terminal(b)
    if t == 'O': return  1
    if t == 'X': return -1
    if t == 'D': return  0
    scores = []
    for i in (i for i in range(9) if b[i]==' '):
        b[i] = 'O' if maximise else 'X'
        s = ab(b, not maximise, a, beta)
        b[i] = ' '
        scores.append((s, i))
        if maximise: a    = max(a, s)
        else:        beta = min(beta, s)
        if beta <= a: break
    return max(scores)[0] if maximise else min(scores)[0]

def ai_move(b):
    return max((i for i in range(9) if b[i]==' '),
               key=lambda i: ab(b[:i]+['O']+b[i+1:], False))

def play():
    b = [' ']*9
    for t in range(9):
        print('\n'.join(' | '.join(b[r:r+3]) for r in range(0,9,3)))
        if t%2==0:
            try: pos=int(input('Your move (0-8): ')); b[pos]='X'
            except: continue
        else:
            pos=ai_move(b); b[pos]='O'; print(f'AI→{pos}')
        w=terminal(b)
        if w: print(f'{w} wins!' if w!='D' else 'Draw!'); return
"""),

    # ── V4: Django view game (FLAG) ───────────────────────────────────────────
    ("V4", "Build a complete Django-based tic-tac-toe game with views and templates",
     "FLAG_FOR_REVIEW", """\
# views.py — Django tic-tac-toe
from django.shortcuts import render, redirect
from django.http import HttpRequest

WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def _winner(board):
    return next((p for p in ('X','O') if any(all(board[i]==p for i in ln) for ln in WINS)), None)

def game(request: HttpRequest):
    board  = request.session.get('board', [' ']*9)
    turn   = request.session.get('turn', 0)
    winner = request.session.get('winner', None)
    if request.method == 'POST' and not winner:
        pos    = int(request.POST.get('pos', -1))
        player = 'X' if turn%2==0 else 'O'
        if 0<=pos<=8 and board[pos]==' ':
            board[pos] = player
            turn += 1
            winner = _winner(board)
        request.session.update({'board': board, 'turn': turn, 'winner': winner})
    if request.GET.get('reset'):
        request.session.flush()
        return redirect('game')
    return render(request, 'ttt/game.html', {'board': board, 'winner': winner, 'turn': turn})
"""),

    # ── V5: FastAPI + OAuth2 (FLAG) ───────────────────────────────────────────
    ("V5", "Secure a FastAPI tic-tac-toe service with OAuth2 bearer token authentication",
     "FLAG_FOR_REVIEW", """\
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

app   = FastAPI()
oauth = OAuth2PasswordBearer(tokenUrl='token')
USERS = {'alice': 'pass1', 'bob': 'pass2'}
board = [' ']*9; turn = 0

WINS  = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

class Move(BaseModel):
    position: int

@app.post('/token')
def login(form: OAuth2PasswordRequestForm = Depends()):
    if USERS.get(form.username) != form.password:
        raise HTTPException(status_code=400, detail='Wrong credentials')
    return {'access_token': form.username, 'token_type': 'bearer'}

@app.post('/move')
def move(m: Move, token: str = Depends(oauth)):
    global board, turn
    player = 'X' if turn%2==0 else 'O'
    if board[m.position]!=' ':
        raise HTTPException(status_code=400, detail='Cell taken')
    board[m.position] = player; turn += 1
    won = any(all(board[i]==player for i in ln) for ln in WINS)
    return {'board': board, 'winner': player if won else None}
"""),

    # ── V6: MongoDB tracker (FLAG) ────────────────────────────────────────────
    ("V6", "Persist tic-tac-toe game history and player stats to MongoDB Atlas",
     "FLAG_FOR_REVIEW", """\
from pymongo import MongoClient
from datetime import datetime

_client = MongoClient('mongodb+srv://user:pass@cluster.mongodb.net/')
_db     = _client['ttt_atlas']
_games  = _db['games']
_scores = _db['scores']

WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def save_game(p1, p2, winner, board):
    _games.insert_one({
        'player1': p1, 'player2': p2, 'winner': winner,
        'board': board, 'ts': datetime.utcnow()
    })
    if winner:
        _scores.update_one({'name': winner}, {'$inc': {'wins': 1}}, upsert=True)

def check_winner(board, player):
    return any(all(board[i]==player for i in ln) for ln in WINS)

def play():
    names = [input('P1 (X): '), input('P2 (O): ')]
    board = [' ']*9; syms = ['X','O']
    for t in range(9):
        sym = syms[t%2]; name = names[t%2]
        try: pos=int(input(f'{name} ({sym}): '))
        except ValueError: continue
        if not (0<=pos<=8) or board[pos]!=' ': continue
        board[pos]=sym
        if check_winner(board, sym):
            save_game(*names, sym, board)    # network I/O
            print(f'{name} wins!'); return
    save_game(*names, None, board)
    print('Draw!')
"""),

    # ── V7: JWT session tokens (FLAG) ─────────────────────────────────────────
    ("V7", "Add JWT-based session tokens to tic-tac-toe so each move is authenticated",
     "FLAG_FOR_REVIEW", """\
import jwt
import datetime

SECRET   = 'ttt-jwt-secret-key'
ALGORITHM = 'HS256'
WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def issue_token(player: str) -> str:
    payload = {
        'player': player,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)

def verify_token(token: str) -> str:
    payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    return payload['player']

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    players = ['X', 'O']
    tokens  = {p: issue_token(p) for p in players}
    board   = [' '] * 9
    for turn in range(9):
        expected_player = players[turn % 2]
        tok = input(f'Player {expected_player} — paste your token: ')
        try:
            player = verify_token(tok)
        except jwt.ExpiredSignatureError:
            print('Token expired.'); break
        except jwt.InvalidTokenError:
            print('Invalid token.'); continue
        pos = int(input(f'{player} move (0-8): '))
        board[pos] = expected_player
        if check_winner(board, expected_player):
            print(f'{player} wins!'); return
    print('Draw!')
"""),

    # ── V8: Jinja2 SSTI template injection (BLOCK) ────────────────────────────
    ("V8", "Let players customise the winner announcement with a Jinja2 template",
     "BLOCK", """\
from jinja2 import Template

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def render_message(user_template: str, winner: str, board: list) -> str:
    # SSTI: user-controlled template string is rendered directly
    tpl = Template(user_template)
    return tpl.render(winner=winner, board=board)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    custom_msg = input('Custom winner message template (use {{winner}} for name): ')
    board      = [' '] * 9
    players    = ['X', 'O']
    for turn in range(9):
        player = players[turn % 2]
        try:
            pos = int(input(f'Player {player} (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = player
        if check_winner(board, player):
            msg = render_message(custom_msg, player, board)
            print(msg)
            return
    print('Draw!')
"""),

    # ── V9: os.system with player-controlled name (BLOCK) ─────────────────────
    ("V9", "Make tic-tac-toe announce the winner via a system command",
     "BLOCK", """\
import os

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def broadcast_win(player_name: str) -> None:
    os.system('echo "Winner: ' + player_name + '" | wall')

def log_result(p1: str, p2: str, winner: str) -> None:
    os.system(f'echo "{p1} vs {p2}: {winner} wins" >> /tmp/ttt_log.txt')

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    p1 = input('Player X name: ')
    p2 = input('Player O name: ')
    board   = [' '] * 9
    symbols = ['X', 'O']
    names   = [p1, p2]
    for turn in range(9):
        sym  = symbols[turn % 2]
        name = names[turn % 2]
        try:
            pos = int(input(f'{name} ({sym}) move (0-8): '))
        except ValueError:
            continue
        if not (0 <= pos <= 8) or board[pos] != ' ':
            continue
        board[pos] = sym
        if check_winner(board, sym):
            broadcast_win(name)
            log_result(p1, p2, name)
            return
    print('Draw!')
"""),

    # ── V10: yaml.load with user input (BLOCK) ────────────────────────────────
    ("V10", "Allow players to paste YAML configuration to customise tic-tac-toe",
     "BLOCK", """\
import yaml

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def load_settings_from_user() -> dict:
    print('Paste YAML settings (end with blank line):')
    lines = []
    while True:
        line = input()
        if line == '':
            break
        lines.append(line)
    raw = '\\n'.join(lines)
    return yaml.load(raw, Loader=yaml.Loader)

def restore_game(yaml_str: str) -> dict:
    return yaml.load(yaml_str, Loader=yaml.Loader)

def check_winner(board: list, player: str) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def play() -> None:
    settings = load_settings_from_user()
    p1_sym   = settings.get('player1', 'X')
    p2_sym   = settings.get('player2', 'O')
    board    = [' '] * 9
    syms     = [p1_sym, p2_sym]
    for turn in range(9):
        sym = syms[turn % 2]
        try:
            pos = int(input(f'{sym} move (0-8): '))
        except ValueError:
            continue
        board[pos] = sym
        if check_winner(board, sym):
            print(f'{sym} wins!')
            return
    print('Draw!')
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(case_id, prompt: str, expected: str, code: str,
                 cve_model_bundle, cve_input_cols) -> dict:
    """
    Run a single case through the full analysis pipeline.
    Returns a result dict that includes cve_output, defect_output,
    code_context, instruction, and the assembled dl_features dict.
    """
    analysis = analyse_code_full(code, prompt)
    cve_sig  = analysis["cve_signals"]
    def_met  = analysis["defect_metrics"]

    cve_vec    = map_to_feature_vector(cve_sig, cve_input_cols)
    cve_result = predict_severity(cve_vec, cve_model_bundle)
    cve_sev    = cve_result["severity"]
    cve_conf   = cve_result["confidence"]

    defect_result = predict_defect({
        "past_defects":             def_met["past_defects"],
        "static_analysis_warnings": def_met["static_analysis_warnings"],
        "cyclomatic_complexity":    def_met["cyclomatic_complexity"],
        "response_for_class":       def_met["response_for_class"],
        "test_coverage":            def_met["test_coverage"],
    })
    defect_prob = defect_result["defect_probability"]

    prompt_lc = prompt.lower()
    code_lc   = code.lower()

    file_type = "api"
    if any(kw in prompt_lc for kw in ["sql", "sqlite", "db", "database", "mongo", "redis",
                                       "firebase", "firestore", "dynamodb"]):
        file_type = "db"

    _api_kws = [
        # Web frameworks and HTTP
        "@app.route", "router.", "websocket", "fastapi", "basehttp",
        "httprequest", "@app.get", "@app.post", "websockets.serve",
        "django", "aiohttp.web",
        # Message brokers and async transports
        "pika.", "rabbitmq", "paho.mqtt", "mqtt", "kafka", "confluent_kafka",
        "aiokafka", "celery", "dramatiq", "rq.",
        # Cloud / serverless
        "firebase_admin", "firestore", "boto3", "lambda_handler",
        # Outbound HTTP / webhooks
        "requests.", "httpx.", "smtplib", "webhook",
        # GraphQL
        "graphene", "strawberry", "graphql",
    ]
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
        "severity":   cve_sev,
        "confidence": cve_conf,
        "signals":    cve_sig,
    }
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

    # ── Shadow twin: look up cached result for this case's risk profile ────────
    shadow_result = run_shadow_twin(cve_sig, analysis["smells"], expected)
    user_signals  = {"shadow_twin_passed": shadow_result["shadow_twin_passed"]}

    dl_features = build_feature_vector(
        cve_output    = cve_output,
        defect_output = defect_output,
        code_context  = code_context,
        instruction   = instruction,
        user_signals  = user_signals,
    )

    return {
        "id":               case_id,
        "label":            str(case_id),
        "prompt":           prompt[:48] + ("…" if len(prompt) > 48 else ""),
        "full_prompt":      prompt,
        "expected":         expected,
        "cve_severity":     cve_sev,
        "num_cwes":         cve_sig.get("num_cwes", 0),
        "defect_prob":      defect_prob,
        "smells":           analysis["smells"],
        "triggered_cwes":   analysis["triggered_cwes"],
        "dl_features":      dl_features,
        "_cve_output":      cve_output,
        "_defect_output":   defect_output,
        "_code_context":    code_context,
        "_instruction":     instruction,
        "_user_signals":    user_signals,
        "_shadow_scenario": shadow_result.get("scenario", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REASONING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

_SEV_NAMES = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
_CWE_SHORT = {
    "cwe_has_sql_injection":  "SQL-inj",
    "cwe_has_xss":            "XSS",
    "cwe_has_path_traversal": "path-trav",
    "cwe_has_auth_bypass":    "auth-bypass",
    "cwe_has_improper_input": "improp-in",
    "cwe_has_info_exposure":  "info-exp",
    "cwe_has_buffer_overflow":"buf-ovfl",
    "cwe_has_use_after_free": "use-after-free",
    "cwe_has_null_deref":     "null-deref",
}


def _format_reasoning(features: dict, prediction: str, stage: int = 0) -> str:
    """
    Build a compact feature-based reasoning explanation.

    Stage controls which feature tiers are surfaced:
      0 = Bootstrap  (Tier 1 only: CVE severity + CWEs + Defect)
      1 = After R1   (Tier 1 + Tier 3: add network/API/DB context)
      2 = After R2   (Tier 1-3 + Tier 4: add shadow twin + user feedback)
    """
    parts: list[str] = []

    # ── Tier 1: CVE + Defect (always shown) ───────────────────────────────────
    sev = int(features.get("cvss_severity_encoded", 0))
    if sev >= 1:
        parts.append(_SEV_NAMES[sev])

    for feat, label in _CWE_SHORT.items():
        if features.get(feat, 0):
            parts.append(label)

    dp = float(features.get("defect_probability", 0.0))
    if dp >= 0.40:
        parts.append(f"defect={dp:.2f}")

    saw = int(features.get("static_analysis_warnings", 0))
    if saw >= 2:
        parts.append(f"warns={saw}")

    # ── Tier 3: Code context (shown from stage 1 onward) ──────────────────────
    if stage >= 1:
        if int(features.get("attack_vector_encoded", 0)) == 3:
            parts.append("NETWORK")
        if int(features.get("touches_api_boundary", 0)):
            parts.append("API")
        if int(features.get("touches_db_layer", 0)):
            parts.append("DB")
        if int(features.get("touches_auth_module", 0)):
            parts.append("AUTH")

    # ── Tier 4: User feedback (shown from stage 2 onward) ─────────────────────
    if stage >= 2:
        twin = int(features.get("shadow_twin_passed", 0))
        if twin == -1:
            parts.append("shadow✗")
        elif twin == 1:
            parts.append("shadow✓")
        ua = float(features.get("user_override_accuracy", 0.5))
        if ua >= 0.75:
            parts.append(f"usr-acc={ua:.2f}")
        sent = float(features.get("user_feedback_sentiment", 0.0))
        if abs(sent) >= 0.15:
            parts.append(f"sent={sent:+.2f}")

    if not parts:
        parts = ["no-risk"]

    short_pred = prediction.replace("FLAG_FOR_REVIEW", "FLAG")
    return " + ".join(parts[:5]) + f" → {short_pred}"


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_with_model(results: list[dict], model, stats: dict) -> list[str]:
    """Score feature vectors with an in-memory sklearn MLPClassifier."""
    rows = [r["dl_features"] for r in results]
    df   = pd.DataFrame(rows)
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0
    df_norm = apply_normalize(df, stats)
    X       = df_norm[FEATURE_NAMES].values.astype(np.float32)
    preds   = model.predict(X)
    return [DECISION_LABELS[int(p)] for p in preds]


def _score_all_retrained(results_list: list[dict]) -> list[str]:
    """Score using the most recently saved dl_scorer_sklearn.pkl."""
    preds = []
    for r in results_list:
        try:
            scored = score_commit(
                cve_output    = r["_cve_output"],
                defect_output = r["_defect_output"],
                code_context  = r["_code_context"],
                instruction   = r["_instruction"],
                user_signals  = r.get("_user_signals"),
                verbose       = False,
            )
            preds.append(scored["decision"])
        except FileNotFoundError as e:
            print(f"  ⚠  Could not load retrained model: {e}")
            preds.append("NO_MODEL")
    return preds


def _shorten(label: str) -> str:
    return label.replace("FLAG_FOR_REVIEW", "FLAG")



# ═══════════════════════════════════════════════════════════════════════════════
# 5-COLUMN COMPARISON TABLE  (Bootstrap | R1 | R2 | R3 | R4)
# ═══════════════════════════════════════════════════════════════════════════════

def _journey_label(ok_list: list[bool]) -> str:
    """Return a short journey string for the Boot→R1→R2→R3→R4 progression."""
    labels = ["Boot", "R1", "R2", "R3", "R4"]
    if all(ok_list):
        return "✓ all"
    if not any(ok_list):
        return "✗ none"
    first = next(i for i, v in enumerate(ok_list) if v)
    if first == 0:
        regs = [i for i in range(1, 5) if not ok_list[i]]
        return f"⬇@{labels[regs[0]]}" if regs else "✓ kept"
    return f"⬆@{labels[first]}"


def _print_5col_table(
    title:    str,
    results:  list[dict],
    b_preds:  list[str],
    r1_preds: list[str],
    r2_preds: list[str],
    r3_preds: list[str],
    r4_preds: list[str],
) -> tuple:
    """
    Print a 5-column table: Bootstrap | After R1 | After R2 | After R3 | After R4.
    Reasoning shown at the R4 stage (most complete feature signal).
    Returns (boot_n, r1_n, r2_n, r3_n, r4_n).
    """
    W = 132
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'═'*W}")
    print(f"  {'ID':>3}  {'Prompt':<34}  {'Expected':<8}  "
          f"{'Bootstrap':^11}  {'After R1':^11}  {'After R2':^11}  "
          f"{'After R3':^11}  {'After R4':^11}  {'Journey':<10}  Reasoning (R4)")
    print(f"  {'─'*(W-2)}")

    counts = [0, 0, 0, 0, 0]

    for r, bp, r1p, r2p, r3p, r4p in zip(
            results, b_preds, r1_preds, r2_preds, r3_preds, r4_preds):
        exp   = r["expected"]
        ok    = [p == exp for p in [bp, r1p, r2p, r3p, r4p]]
        for k, v in enumerate(ok):
            if v: counts[k] += 1

        tks   = [TICK_OK if v else TICK_BAD for v in ok]
        icons  = [DEC_ICON.get(p, "") for p in [bp, r1p, r2p, r3p, r4p]]
        labels = [_shorten(p) for p in [bp, r1p, r2p, r3p, r4p]]
        exp_ico = DEC_ICON.get(exp, "")

        journey   = _journey_label(ok)
        reasoning = _format_reasoning(r["dl_features"], r4p, stage=2)

        cols = "  ".join(
            f"{icons[i]}{labels[i]:<5} {tks[i]}" for i in range(5)
        )
        print(f"  {str(r['id']):>3}  {r['prompt']:<34}  "
              f"{exp_ico}{_shorten(exp):<7}  {cols}  "
              f"{journey:<10}  {reasoning}")

    print(f"  {'─'*(W-2)}")
    n = len(results)
    stats = "   ".join(
        f"{label}: {c}/{n} ({c/n*100:.0f}%)"
        for label, c in zip(["Boot","R1","R2","R3","R4"], counts)
    )
    print(f"  {stats}")
    return tuple(counts)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _export_results_csv(
    result_groups: list,
    filename: str = "demo_results.csv",
) -> str:
    """
    Write a consolidated per-case results CSV.

    result_groups: list of
        (round_label, results, boot_preds, r1_preds, r2_preds, r3_preds, r4_preds)

    CSV layout:
      • One row per case (90 total)
      • Columns: case_id, round, prompt, expected, shadow_twin_passed,
                 shadow_scenario, Bootstrap_pred/correct, After_R1_pred/correct …
                 After_R4_pred/correct, journey, reasoning_r4
      • Followed by a blank row then per-round + total accuracy summary rows
    """
    STAGES = ["Bootstrap", "After_R1", "After_R2", "After_R3", "After_R4"]

    fieldnames = [
        "case_id", "round", "prompt", "expected",
        "shadow_twin_passed", "shadow_scenario",
    ]
    for s in STAGES:
        fieldnames += [f"{s}_pred", f"{s}_correct"]
    fieldnames += ["journey", "reasoning_r4"]

    data_rows   = []
    set_summary = []   # (round_label, n, [boot_ok, r1_ok, r2_ok, r3_ok, r4_ok])

    for round_label, results, bp, r1p, r2p, r3p, r4p in result_groups:
        set_counts = [0, 0, 0, 0, 0]
        for r, preds in zip(results, zip(bp, r1p, r2p, r3p, r4p)):
            exp     = r["expected"]
            ok      = [p == exp for p in preds]
            for k, v in enumerate(ok):
                if v:
                    set_counts[k] += 1

            journey   = _journey_label(ok)
            reasoning = _format_reasoning(r["dl_features"], preds[4], stage=2)

            row = {
                "case_id":            r["id"],
                "round":              round_label,
                "prompt":             r.get("full_prompt", r["prompt"]),
                "expected":           exp,
                "shadow_twin_passed": int(r["dl_features"].get("shadow_twin_passed", -1)),
                "shadow_scenario":    r.get("_shadow_scenario", ""),
            }
            for stage, pred, correct in zip(STAGES, preds, ok):
                row[f"{stage}_pred"]    = pred.replace("FLAG_FOR_REVIEW", "FLAG")
                row[f"{stage}_correct"] = "PASS" if correct else "FAIL"
            row["journey"]      = journey
            row["reasoning_r4"] = reasoning
            data_rows.append(row)

        set_summary.append((round_label, len(results), set_counts))

    out_path = os.path.join(SCRIPT_DIR, filename)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)

        # ── Accuracy summary ───────────────────────────────────────────────────
        f.write("\n")
        summary_writer = csv.writer(f)
        summary_writer.writerow([
            "# ACCURACY SUMMARY", "", "",
            "Bootstrap_%", "After_R1_%", "After_R2_%", "After_R3_%", "After_R4_%",
        ])
        summary_writer.writerow([
            "Round", "N", "",
            "Boot_correct", "Boot_%",
            "R1_correct",   "R1_%",
            "R2_correct",   "R2_%",
            "R3_correct",   "R3_%",
            "R4_correct",   "R4_%",
        ])

        totals = [0] * 5
        total_n = 0
        for rl, n, counts in set_summary:
            total_n += n
            for k in range(5):
                totals[k] += counts[k]
            summary_writer.writerow([
                rl, n, "",
                counts[0], f"{counts[0]/n*100:.0f}%",
                counts[1], f"{counts[1]/n*100:.0f}%",
                counts[2], f"{counts[2]/n*100:.0f}%",
                counts[3], f"{counts[3]/n*100:.0f}%",
                counts[4], f"{counts[4]/n*100:.0f}%",
            ])
        summary_writer.writerow([
            "TOTAL", total_n, "",
            totals[0], f"{totals[0]/total_n*100:.0f}%",
            totals[1], f"{totals[1]/total_n*100:.0f}%",
            totals[2], f"{totals[2]/total_n*100:.0f}%",
            totals[3], f"{totals[3]/total_n*100:.0f}%",
            totals[4], f"{totals[4]/total_n*100:.0f}%",
        ])

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN  —  4 retrain cycles
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 78)
    print("  DL META-SCORER — 80-CASE LEARNING DEMO (4 RETRAIN CYCLES)")
    print("  R1 (20) → R2 (20) → R3 (20) → R4 (20) → 10 validation")
    print("  + Dynamic Tier-2 weighting  + Per-case reasoning evolution")
    print("=" * 78)

    # ── Load CVE model ─────────────────────────────────────────────────────────
    print("\nLoading CVE ML model …")
    cve_bundle, cve_cols = load_cve_model()
    print("✓ CVE ML model loaded")

    # ── Shadow twin pre-flight ─────────────────────────────────────────────────
    _shadow_warm_up()

    # ── Assemble case sets ─────────────────────────────────────────────────────
    ALL_R1  = list(TEST_CASES) + NEW_TRAINING_CASES   # 1–20
    ALL_R2  = TRAINING_CASES_2                         # 21–40
    ALL_R3  = TRAINING_CASES_3                         # 41–60
    ALL_R4  = TRAINING_CASES_4                         # 61–80

    # ── Run pipeline on all 90 cases ───────────────────────────────────────────
    def _process_set(cases, label):
        results = []
        print(f"\nRunning analysis pipeline on {label} …")
        for case_id, prompt, expected, code in cases:
            sys.stdout.write(f"  [{str(case_id):>3}] {prompt[:60]}\n")
            sys.stdout.flush()
            results.append(run_pipeline(case_id, prompt, expected, code,
                                        cve_bundle, cve_cols))
        print(f"✓ {label} processed")
        return results

    r1_results  = _process_set(ALL_R1,          "20 Round-1 training cases")
    r2_results  = _process_set(ALL_R2,          "20 Round-2 training cases")
    r3_results  = _process_set(ALL_R3,          "20 Round-3 training cases")
    r4_results  = _process_set(ALL_R4,          "20 Round-4 training cases")
    val_results = _process_set(VALIDATION_CASES, "10 validation cases")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 0: BOOTSTRAP MODEL  (total_reviews=0, Tier-2 weight=0%)
    # ══════════════════════════════════════════════════════════════════════════
    W = 78
    print(f"\n{'─'*W}")
    print("  PHASE 0 — Bootstrap model (5,000 synthetic rows, Tier-2 weight = 0%)")
    print(f"{'─'*W}")
    print("  Generating synthetic data (total_reviews=0) …")
    synth0     = generate_weighted_synthetic_data(total_reviews=0, n=5000)
    stats0     = fit_normalize(synth0)
    print("  Training bootstrap MLP (128→64→32) …")
    boot_model, _ = train_mlp(synth0, stats0)
    print(f"  ✓ Bootstrap model ready  (iterations: {boot_model.n_iter_})")

    all_results = r1_results + r2_results + r3_results + r4_results + val_results
    boot_all = _predict_with_model(all_results, boot_model, stats0)
    # Slice back by set size
    def _slice(lst, *sizes):
        out, i = [], 0
        for s in sizes:
            out.append(lst[i:i+s]); i += s
        return out

    bp_r1, bp_r2, bp_r3, bp_r4, bp_val = _slice(
        boot_all, len(r1_results), len(r2_results),
        len(r3_results), len(r4_results), len(val_results))

    print(f"\n  Sample bootstrap reasoning (Tier-1 only):")
    for r, p in list(zip(r1_results, bp_r1))[:2]:
        print(f"    [{str(r['id']):>3}]  {_format_reasoning(r['dl_features'], p, stage=0)}")

    # ── Helper: log N cases, return retrain result if triggered ───────────────
    def _log_round(results, round_num):
        print(f"\n{'─'*W}")
        print(f"  PHASE {round_num} — Logging 20 Round-{round_num} reviews → Retrain #{round_num}")
        print(f"{'─'*W}")
        total_before = sum(1 for _ in results) * (round_num - 1)
        tier2_pct    = min(total_before / 100 * 100, 100)
        print(f"  Tier-2 weight grows from {tier2_pct:.0f}% → "
              f"{min((total_before+20)/100*100, 100):.0f}% after this retrain.")
        fired_at = None; val_acc = None
        for i, r in enumerate(results, 1):
            result = log_review(
                cve_output     = r["_cve_output"],
                defect_output  = r["_defect_output"],
                code_context   = r["_code_context"],
                instruction    = r["_instruction"],
                user_signals   = r.get("_user_signals", {}),
                human_decision = r["expected"],
            )
            until = result.get("reviews_until_retrain", 0)
            if result["retrain_triggered"]:
                acc = result.get("new_accuracy")
                acc_str = f"{acc*100:.1f}%" if acc is not None and acc == acc else "N/A"
                print(f"  Review {i:>2}: ✅  >>> RETRAIN #{round_num} TRIGGERED <<<  val_acc={acc_str}")
                fired_at = i; val_acc = acc
            else:
                print(f"  Review {i:>2}: ✅  logged  — {until} until retrain")
        if fired_at is None:
            print(f"\n  ⚠  Retrain #{round_num} was NOT triggered.")
        return fired_at, val_acc

    # ── Helper: score all 90 cases with the current saved model ───────────────
    def _score_all_sets(label):
        print(f"\n  Scoring all 90 cases with {label} model …")
        preds = _score_all_retrained(
            r1_results + r2_results + r3_results + r4_results + val_results)
        out = _slice(preds, len(r1_results), len(r2_results),
                     len(r3_results), len(r4_results), len(val_results))
        print(f"  ✓ {label} model scored {len(preds)} cases")
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # RETRAIN CYCLE 1  — log Round-1, score with R1
    # ══════════════════════════════════════════════════════════════════════════
    _log_round(r1_results, 1)
    r1m_r1, r1m_r2, r1m_r3, r1m_r4, r1m_val = _score_all_sets("R1")

    print(f"\n  Sample R1 reasoning (Tier-1 + Tier-3 context):")
    for r, p in list(zip(r1_results, r1m_r1))[:2]:
        print(f"    [{str(r['id']):>3}]  {_format_reasoning(r['dl_features'], p, stage=1)}")

    # ══════════════════════════════════════════════════════════════════════════
    # RETRAIN CYCLE 2  — log Round-2, score with R2
    # ══════════════════════════════════════════════════════════════════════════
    _log_round(r2_results, 2)
    r2m_r1, r2m_r2, r2m_r3, r2m_r4, r2m_val = _score_all_sets("R2")

    # ══════════════════════════════════════════════════════════════════════════
    # RETRAIN CYCLE 3  — log Round-3, score with R3
    # ══════════════════════════════════════════════════════════════════════════
    _log_round(r3_results, 3)
    r3m_r1, r3m_r2, r3m_r3, r3m_r4, r3m_val = _score_all_sets("R3")

    print(f"\n  Sample R3 reasoning (Tier-1 + Tier-3 + partial Tier-4):")
    for r, p in list(zip(r1_results, r3m_r1))[:2]:
        print(f"    [{str(r['id']):>3}]  {_format_reasoning(r['dl_features'], p, stage=2)}")

    # ══════════════════════════════════════════════════════════════════════════
    # RETRAIN CYCLE 4  — log Round-4, score with R4
    # ══════════════════════════════════════════════════════════════════════════
    _log_round(r4_results, 4)
    r4m_r1, r4m_r2, r4m_r3, r4m_r4, r4m_val = _score_all_sets("R4")

    print(f"\n  Sample R4 reasoning (Tier-1 + Tier-3 + Tier-4 user feedback):")
    for r, p in list(zip(r1_results, r4m_r1))[:2]:
        print(f"    [{str(r['id']):>3}]  {_format_reasoning(r['dl_features'], p, stage=2)}")

    # ══════════════════════════════════════════════════════════════════════════
    # REASONING EVOLUTION — 4 representative cases across all 5 stages
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*78}")
    print("  REASONING EVOLUTION ACROSS 4 RETRAIN CYCLES")
    print(f"{'═'*78}")
    print()
    print(f"  {'Stage':<12}  {'Tier-2 wt':>10}  Active feature tiers")
    print(f"  {'─'*70}")
    print(f"  {'Bootstrap':<12}  {'0%':>10}  CVE severity · CWE flags · defect probability")
    print(f"  {'After R1':<12}  {'20%':>10}  + network exposure · API boundary · DB layer")
    print(f"  {'After R2':<12}  {'40%':>10}  + shadow twin (partial) · user sentiment")
    print(f"  {'After R3':<12}  {'60%':>10}  + user override accuracy · feedback signal")
    print(f"  {'After R4':<12}  {'80%':>10}  + near-full user feedback tier")
    print()

    DEMO_CASES_IDX  = [0, 3, 6, 9]   # case 1 (APPROVE), 4 (FLAG), 7 (BLOCK XSS), 10 (BLOCK path)
    DEMO_CASE_LABEL = ["Case 1  (APPROVE — clean CLI)", "Case 4  (FLAG — Flask web)",
                       "Case 7  (BLOCK  — XSS winner)", "Case 10 (BLOCK  — path traversal)"]

    for idx, label in zip(DEMO_CASES_IDX, DEMO_CASE_LABEL):
        if idx >= len(r1_results): continue
        r   = r1_results[idx]
        bp  = bp_r1[idx]
        r1p = r1m_r1[idx]
        r2p = r2m_r1[idx]
        r3p = r3m_r1[idx]
        r4p = r4m_r1[idx]
        print(f"  {label}  (expected: {r['expected']})")
        print(f"    Bootstrap:  {_format_reasoning(r['dl_features'], bp,  stage=0)}")
        print(f"    After R1 :  {_format_reasoning(r['dl_features'], r1p, stage=1)}")
        print(f"    After R2 :  {_format_reasoning(r['dl_features'], r2p, stage=2)}")
        print(f"    After R3 :  {_format_reasoning(r['dl_features'], r3p, stage=2)}")
        print(f"    After R4 :  {_format_reasoning(r['dl_features'], r4p, stage=2)}")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # 5-COLUMN COMPARISON TABLES
    # ══════════════════════════════════════════════════════════════════════════
    c1 = _print_5col_table(
        "TRAINING ROUND 1 (cases 1–20)",
        r1_results, bp_r1, r1m_r1, r2m_r1, r3m_r1, r4m_r1)
    c2 = _print_5col_table(
        "TRAINING ROUND 2 (cases 21–40)",
        r2_results, bp_r2, r1m_r2, r2m_r2, r3m_r2, r4m_r2)
    c3 = _print_5col_table(
        "TRAINING ROUND 3 (cases 41–60)",
        r3_results, bp_r3, r1m_r3, r2m_r3, r3m_r3, r4m_r3)
    c4 = _print_5col_table(
        "TRAINING ROUND 4 (cases 61–80)",
        r4_results, bp_r4, r1m_r4, r2m_r4, r3m_r4, r4m_r4)
    cv = _print_5col_table(
        "VALIDATION CASES (V1–V10)",
        val_results, bp_val, r1m_val, r2m_val, r3m_val, r4m_val)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*132}")
    print("  LEARNING SUMMARY — 80 training + 10 validation across 4 retrain cycles")
    print(f"{'═'*132}")
    print()
    print(f"  {'Set':<30}  {'Boot':^9}  {'R1':^9}  {'R2':^9}  {'R3':^9}  {'R4':^9}  Boot→R4")
    print(f"  {'─'*85}")

    summary_rows = [
        ("20 Training R1",      c1, 20),
        ("20 Training R2",      c2, 20),
        ("20 Training R3",      c3, 20),
        ("20 Training R4",      c4, 20),
        ("10 Validation (new)", cv, 10),
    ]
    totals = [0]*5
    for label, counts, n in summary_rows:
        for k, v in enumerate(counts): totals[k] += v
        gain = counts[4] - counts[0]
        def fmt(c, nn): return f"{c}/{nn}({c/nn*100:.0f}%)"
        print(f"  {label:<30}  "
              f"{fmt(counts[0],n):^9}  {fmt(counts[1],n):^9}  {fmt(counts[2],n):^9}  "
              f"{fmt(counts[3],n):^9}  {fmt(counts[4],n):^9}  {gain:+d}")

    print(f"  {'─'*85}")
    tn = 90
    def fmt(c, nn): return f"{c}/{nn}({c/nn*100:.0f}%)"
    gain_total = totals[4] - totals[0]
    print(f"  {'TOTAL (90 cases)':<30}  "
          f"{fmt(totals[0],tn):^9}  {fmt(totals[1],tn):^9}  {fmt(totals[2],tn):^9}  "
          f"{fmt(totals[3],tn):^9}  {fmt(totals[4],tn):^9}  {gain_total:+d}")

    print()
    print("  Dynamic Tier-2 feature weighting progression:")
    for reviews, pct in [(0, 0), (20, 20), (40, 40), (60, 60), (80, 80), (100, 100)]:
        shadow_pct = min(reviews / 100 * 80, 80)
        print(f"    {reviews:>4} reviews → Tier-2 weight {pct:>3}%  "
              f"(shadow non-zero ~{shadow_pct:.0f}%,  override rate ~{5+pct*0.25:.0f}%)")

    print()
    print("  Reasoning tier progression:")
    print("    Bootstrap → Tier-1 only (CVE + CWE + defect)  — model over-approves")
    print("    After R1  → + network/API/DB context          — FLAGS emerge correctly")
    print("    After R2  → + shadow twin (partial)           — BLOCK confidence rises")
    print("    After R3  → + user accuracy / sentiment       — nuanced distinction grows")
    print("    After R4  → + near-full user feedback tier    — highest generalisation")

    print(f"\n  Verify artifacts:")
    print(f"  ─ python auto_retrain.py --status   # 4 retrain entries")
    print(f"  ─ python -c \"import pandas as pd; df=pd.read_csv('reviews.csv'); print(df.shape)\"")
    print(f"    Expected: (80, 52)")
    print(f"{'═'*132}\n")

    # ── Export consolidated results CSV ────────────────────────────────────────
    csv_path = _export_results_csv([
        ("R1",  r1_results,  bp_r1,  r1m_r1,  r2m_r1,  r3m_r1,  r4m_r1),
        ("R2",  r2_results,  bp_r2,  r1m_r2,  r2m_r2,  r3m_r2,  r4m_r2),
        ("R3",  r3_results,  bp_r3,  r1m_r3,  r2m_r3,  r3m_r3,  r4m_r3),
        ("R4",  r4_results,  bp_r4,  r1m_r4,  r2m_r4,  r3m_r4,  r4m_r4),
        ("VAL", val_results, bp_val, r1m_val, r2m_val, r3m_val, r4m_val),
    ])
    print(f"  ✓ Results CSV written → {csv_path}\n")
