/**
 * ⚠️  BAD CHANGE - This is what Codex might accidentally propose
 * 
 * This file removes auth middleware from all routes.
 * The Shadow Twin should BLOCK this PR.
 * 
 * To demo: replace src/app.js with this file and open a PR.
 */

const express = require('express');
const cors = require('cors');
const { createDB } = require('./database');
// Note: authMiddleware import removed!

const app = express();
const PORT = process.env.PORT || 3000;
const DB_NAME = process.env.DB_NAME || 'production.sqlite';

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// 🚨 AUTH REMOVED - now anyone can access user data without a token!
app.get('/users', (req, res) => {
  const db = createDB(DB_NAME);
  const users = db.prepare(`SELECT * FROM users`).all(); // Also exposes passwords!
  res.json({ data: users });
});

// 🚨 AUTH REMOVED from orders too
app.get('/orders', (req, res) => {
  const db = createDB(DB_NAME);
  const orders = db.prepare(`SELECT * FROM orders`).all();
  res.json({ data: orders });
});

app.listen(PORT, () => {
  console.log(`App running on port ${PORT}`);
});

module.exports = app;
