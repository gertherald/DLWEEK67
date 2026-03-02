/**
 * bad-ttt-no-validation.js — BAD VERSION: input validation removed
 *
 * Simulates a Codex-generated change that removes input sanitization from
 * /game/start — accepting arbitrary player names including XSS payloads,
 * SQL injection strings, and other malicious input without rejection.
 *
 * Shadow twin tests will FAIL: <script> and SQL-injection player names
 * return 201 instead of 400.
 */

const express = require('express');
const cors    = require('cors');
const { createDB }       = require('../src/database');
const { authMiddleware } = require('../src/auth');

const app     = express();
const PORT    = process.env.PORT    || 3000;
const DB_NAME = process.env.DB_NAME || 'production.sqlite';

app.use(cors());
app.use(express.json());

function initTTT(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS games (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      player_x       TEXT    NOT NULL,
      player_o       TEXT    DEFAULT NULL,
      board          TEXT    DEFAULT '---------',
      current_player TEXT    DEFAULT 'X',
      status         TEXT    DEFAULT 'active',
      winner         TEXT    DEFAULT NULL,
      created_at     TEXT    DEFAULT (datetime('now'))
    );
  `);
}

function checkWinner(board) {
  const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (const [a, b, c] of wins) {
    if (board[a] !== '-' && board[a] === board[b] && board[b] === board[c]) return board[a];
  }
  return null;
}

app.get('/health', (req, res) => {
  res.json({ status: 'ok', app: 'TicTacToe Game Server (NO VALIDATION)', timestamp: new Date().toISOString() });
});

app.post('/auth/login', (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required' });
  const db = createDB(DB_NAME);
  const user = db.prepare(`SELECT * FROM users WHERE email = ?`).get(email);
  if (!user) return res.status(404).json({ error: 'User not found' });
  const session = db.prepare(
    `SELECT token FROM sessions WHERE user_id = ? AND expires_at > datetime('now') LIMIT 1`
  ).get(user.id);
  if (!session) return res.status(401).json({ error: 'No active session' });
  res.json({ token: session.token });
});

// BAD: auth is present but all input validation has been removed
app.post('/game/start', authMiddleware, (req, res) => {
  const { player_name } = req.body;

  // REMOVED: length check
  // REMOVED: HTML/script tag check  ← XSS vulnerability
  // REMOVED: SQL injection check    ← injection vulnerability
  // Any player_name is blindly accepted

  if (!player_name) return res.status(400).json({ error: 'player_name required' });

  const db = createDB(DB_NAME);
  initTTT(db);
  const result = db.prepare(`INSERT INTO games (player_x) VALUES (?)`).run(String(player_name));
  res.status(201).json({ game_id: result.lastInsertRowid, board: '---------', status: 'active' });
});

// Move endpoint — position validation kept (only name validation is broken)
app.post('/game/:id/move', authMiddleware, (req, res) => {
  const { position } = req.body;
  const game_id = parseInt(req.params.id);
  if (position === undefined || position < 0 || position > 8)
    return res.status(400).json({ error: 'position must be 0–8' });
  const db   = createDB(DB_NAME);
  initTTT(db);
  const game = db.prepare(`SELECT * FROM games WHERE id = ?`).get(game_id);
  if (!game) return res.status(404).json({ error: 'Game not found' });
  if (game.status !== 'active') return res.status(400).json({ error: 'Game is finished' });
  const board = game.board.split('');
  if (board[position] !== '-') return res.status(400).json({ error: 'Cell occupied' });
  board[position] = game.current_player;
  const winner = checkWinner(board);
  const status = winner ? 'finished' : !board.includes('-') ? 'draw' : 'active';
  db.prepare(`UPDATE games SET board=?, current_player=?, status=?, winner=? WHERE id=?`)
    .run(board.join(''), game.current_player === 'X' ? 'O' : 'X', status, winner, game_id);
  res.json({ board: board.join(''), status, winner });
});

app.get('/game/:id/state', authMiddleware, (req, res) => {
  const db   = createDB(DB_NAME);
  initTTT(db);
  const game = db.prepare(`SELECT * FROM games WHERE id = ?`).get(parseInt(req.params.id));
  if (!game) return res.status(404).json({ error: 'Game not found' });
  res.json({ data: game });
});

app.get('/leaderboard', authMiddleware, (req, res) => {
  const db    = createDB(DB_NAME);
  initTTT(db);
  const stats = db.prepare(
    `SELECT player_x AS player, COUNT(*) AS games,
            SUM(CASE WHEN winner='X' THEN 1 ELSE 0 END) AS wins
     FROM games WHERE status='finished' GROUP BY player_x`
  ).all();
  res.json({ data: stats });
});

app.listen(PORT, () => {
  console.log(`\n⚠️  TicTacToe Server (NO VALIDATION) running on http://localhost:${PORT}`);
});

module.exports = app;
