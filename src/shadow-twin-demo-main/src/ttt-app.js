/**
 * ttt-app.js — Secure Tic-Tac-Toe Game Server (Shadow Twin Baseline)
 *
 * This is the GOOD version of the server.
 * All routes enforce auth, all inputs are validated, all SQL uses parameterized queries.
 * The shadow-ttt-tests.js suite should PASS against this server.
 */

const express = require('express');
const cors    = require('cors');
const { createDB }       = require('./database');
const { authMiddleware } = require('./auth');

const app     = express();
const PORT    = process.env.PORT    || 3000;
const DB_NAME = process.env.DB_NAME || 'production.sqlite';

app.use(cors());
app.use(express.json());

// ── TTT table init ─────────────────────────────────────────────────────────────
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

// ── Winner checker ─────────────────────────────────────────────────────────────
function checkWinner(board) {
  const wins = [
    [0,1,2],[3,4,5],[6,7,8],   // rows
    [0,3,6],[1,4,7],[2,5,8],   // cols
    [0,4,8],[2,4,6],           // diagonals
  ];
  for (const [a, b, c] of wins) {
    if (board[a] !== '-' && board[a] === board[b] && board[b] === board[c]) {
      return board[a];
    }
  }
  return null;
}

// ── GET /health (public) ───────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    app: 'TicTacToe Game Server',
    timestamp: new Date().toISOString(),
    db: DB_NAME,
  });
});

// ── POST /auth/login (public) ──────────────────────────────────────────────────
app.post('/auth/login', (req, res) => {
  const { email } = req.body;
  if (!email) return res.status(400).json({ error: 'Email required' });

  const db   = createDB(DB_NAME);
  const user = db.prepare(`SELECT * FROM users WHERE email = ?`).get(email);
  if (!user)  return res.status(404).json({ error: 'User not found' });

  const session = db.prepare(`
    SELECT token FROM sessions
    WHERE user_id = ? AND expires_at > datetime('now')
    LIMIT 1
  `).get(user.id);
  if (!session) return res.status(401).json({ error: 'No active session' });

  res.json({ message: 'Login successful', token: session.token,
             user: { id: user.id, name: user.name, email: user.email } });
});

// ── POST /game/start (auth required) ──────────────────────────────────────────
app.post('/game/start', authMiddleware, (req, res) => {
  const { player_name } = req.body;

  if (!player_name || typeof player_name !== 'string') {
    return res.status(400).json({ error: 'player_name is required' });
  }
  if (player_name.trim().length === 0 || player_name.length > 50) {
    return res.status(400).json({ error: 'player_name must be 1–50 characters' });
  }
  // Block HTML/script injection
  if (/<[^>]*>/.test(player_name)) {
    return res.status(400).json({ error: 'HTML tags are not allowed in player_name' });
  }
  // Block SQL injection patterns
  if (/['";]|(--)|(\bOR\b)|(\bDROP\b)|(\bSELECT\b)|(\bINSERT\b)/i.test(player_name)) {
    return res.status(400).json({ error: 'Invalid characters in player_name' });
  }

  const db = createDB(DB_NAME);
  initTTT(db);
  const result = db.prepare(
    `INSERT INTO games (player_x) VALUES (?)`
  ).run(player_name.trim());

  res.status(201).json({
    game_id:        result.lastInsertRowid,
    board:          '---------',
    current_player: 'X',
    status:         'active',
  });
});

// ── POST /game/:id/move (auth required) ───────────────────────────────────────
app.post('/game/:id/move', authMiddleware, (req, res) => {
  const { position } = req.body;
  const game_id = parseInt(req.params.id);

  if (position === undefined || position === null ||
      !Number.isInteger(position) || position < 0 || position > 8) {
    return res.status(400).json({ error: 'position must be an integer 0–8' });
  }

  const db   = createDB(DB_NAME);
  initTTT(db);
  const game = db.prepare(`SELECT * FROM games WHERE id = ?`).get(game_id);
  if (!game) return res.status(404).json({ error: 'Game not found' });
  if (game.status !== 'active') {
    return res.status(400).json({ error: 'Game is already finished' });
  }

  const board = game.board.split('');
  if (board[position] !== '-') {
    return res.status(400).json({ error: 'Cell is already occupied' });
  }

  board[position] = game.current_player;
  const winner     = checkWinner(board);
  const boardStr   = board.join('');
  const next       = game.current_player === 'X' ? 'O' : 'X';
  const status     = winner          ? 'finished'
                   : !board.includes('-') ? 'draw'
                   : 'active';

  db.prepare(`
    UPDATE games SET board=?, current_player=?, status=?, winner=? WHERE id=?
  `).run(boardStr, next, status, winner, game_id);

  res.json({ board: boardStr, current_player: next, status, winner });
});

// ── GET /game/:id/state (auth required) ───────────────────────────────────────
app.get('/game/:id/state', authMiddleware, (req, res) => {
  const db   = createDB(DB_NAME);
  initTTT(db);
  const game = db.prepare(`SELECT * FROM games WHERE id = ?`).get(parseInt(req.params.id));
  if (!game) return res.status(404).json({ error: 'Game not found' });
  res.json({ data: game });
});

// ── GET /leaderboard (auth required) ──────────────────────────────────────────
app.get('/leaderboard', authMiddleware, (req, res) => {
  const db   = createDB(DB_NAME);
  initTTT(db);
  const stats = db.prepare(`
    SELECT player_x AS player,
           COUNT(*)                                      AS games,
           SUM(CASE WHEN winner = 'X' THEN 1 ELSE 0 END) AS wins
    FROM   games
    WHERE  status = 'finished'
    GROUP  BY player_x
    ORDER  BY wins DESC
  `).all();
  res.json({ data: stats });
});

// ── Start server ───────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🎮 TicTacToe Game Server running on http://localhost:${PORT}`);
  console.log(`📦 Database: ${DB_NAME}`);
  console.log(`\nEndpoints:`);
  console.log(`  GET  /health           — Health check (public)`);
  console.log(`  POST /auth/login       — Login (public)`);
  console.log(`  POST /game/start       — Start new game (auth required)`);
  console.log(`  POST /game/:id/move    — Make a move  (auth required)`);
  console.log(`  GET  /game/:id/state   — Game state   (auth required)`);
  console.log(`  GET  /leaderboard      — Win stats    (auth required)`);
});

module.exports = app;
