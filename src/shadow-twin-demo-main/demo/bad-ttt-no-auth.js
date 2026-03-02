/**
 * bad-ttt-no-auth.js — BAD VERSION: auth middleware removed
 *
 * Simulates a Codex-generated change that accidentally removes authentication
 * from all protected routes. All endpoints now respond without checking the
 * Authorization header.
 *
 * Shadow twin tests will FAIL: /game/start and /leaderboard return 200/201
 * instead of 401 when no token is provided.
 */

const express = require('express');
const cors    = require('cors');
const { createDB } = require('../src/database');
// authMiddleware intentionally NOT imported — this is the bad change

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
  res.json({ status: 'ok', app: 'TicTacToe Game Server (NO AUTH)', timestamp: new Date().toISOString() });
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

// BAD: no authMiddleware — anyone can start a game without a token
app.post('/game/start', (req, res) => {
  const { player_name } = req.body;
  if (!player_name) return res.status(400).json({ error: 'player_name required' });

  // Input validation still present (only auth is broken in this scenario)
  if (/<[^>]*>/.test(player_name)) return res.status(400).json({ error: 'Invalid player_name' });
  if (/['";]|(--)|(\bOR\b)|(\bDROP\b)|(\bSELECT\b)/i.test(player_name))
    return res.status(400).json({ error: 'Invalid characters in player_name' });

  const db = createDB(DB_NAME);
  initTTT(db);
  const result = db.prepare(`INSERT INTO games (player_x) VALUES (?)`).run(player_name.trim());
  res.status(201).json({ game_id: result.lastInsertRowid, board: '---------', status: 'active' });
});

// BAD: no authMiddleware on move endpoint
app.post('/game/:id/move', (req, res) => {
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

// BAD: no authMiddleware on state endpoint
app.get('/game/:id/state', (req, res) => {
  const db   = createDB(DB_NAME);
  initTTT(db);
  const game = db.prepare(`SELECT * FROM games WHERE id = ?`).get(parseInt(req.params.id));
  if (!game) return res.status(404).json({ error: 'Game not found' });
  res.json({ data: game });
});

// BAD: no authMiddleware on leaderboard
app.get('/leaderboard', (req, res) => {
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
  console.log(`\n⚠️  TicTacToe Server (NO AUTH) running on http://localhost:${PORT}`);
});

module.exports = app;
