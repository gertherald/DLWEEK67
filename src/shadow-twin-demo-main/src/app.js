const express = require('express');
const cors = require('cors');
const { createDB } = require('./database');
const { authMiddleware } = require('./auth');

const app = express();
const PORT = process.env.PORT || 3000;
const DB_NAME = process.env.DB_NAME || 'production.sqlite';

app.use(cors());
app.use(express.json());

// ─── Health Check (public) ────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    app: 'ACME Production App',
    timestamp: new Date().toISOString(),
    db: DB_NAME
  });
});

// ─── AUTH endpoint ────────────────────────────────────────────────────────────
app.post('/auth/login', (req, res) => {
  const { email } = req.body;

  if (!email) {
    return res.status(400).json({ error: 'Email required' });
  }

  const db = createDB(DB_NAME);
  const user = db.prepare(`SELECT * FROM users WHERE email = ?`).get(email);

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  // Get their active session token
  const session = db.prepare(`
    SELECT token FROM sessions 
    WHERE user_id = ? AND expires_at > datetime('now')
    LIMIT 1
  `).get(user.id);

  if (!session) {
    return res.status(401).json({ error: 'No active session' });
  }

  res.json({
    message: 'Login successful',
    token: session.token,
    user: { id: user.id, name: user.name, email: user.email, role: user.role }
  });
});

// ─── USERS endpoint (protected) ───────────────────────────────────────────────
app.get('/users', authMiddleware, (req, res) => {
  const db = createDB(DB_NAME);
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const offset = (page - 1) * limit;

  const users = db.prepare(`
    SELECT id, name, email, role, created_at 
    FROM users 
    LIMIT ? OFFSET ?
  `).all(limit, offset);

  const total = db.prepare(`SELECT COUNT(*) as count FROM users`).get();

  res.json({
    data: users,
    pagination: { page, limit, total: total.count }
  });
});

app.get('/users/:id', authMiddleware, (req, res) => {
  const db = createDB(DB_NAME);
  const user = db.prepare(`
    SELECT id, name, email, role, created_at FROM users WHERE id = ?
  `).get(req.params.id);

  if (!user) return res.status(404).json({ error: 'User not found' });
  res.json({ data: user });
});

// ─── ORDERS endpoint (protected) ─────────────────────────────────────────────
app.get('/orders', authMiddleware, (req, res) => {
  const db = createDB(DB_NAME);
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const offset = (page - 1) * limit;

  const orders = db.prepare(`
    SELECT o.*, u.name as user_name, u.email as user_email
    FROM orders o
    JOIN users u ON o.user_id = u.id
    LIMIT ? OFFSET ?
  `).all(limit, offset);

  const total = db.prepare(`SELECT COUNT(*) as count FROM orders`).get();
  const revenue = db.prepare(`SELECT SUM(amount) as total FROM orders WHERE status = 'completed'`).get();

  res.json({
    data: orders,
    pagination: { page, limit, total: total.count },
    stats: { total_revenue: revenue.total || 0 }
  });
});

app.post('/orders', authMiddleware, (req, res) => {
  const { product, amount } = req.body;

  if (!product || !amount) {
    return res.status(400).json({ error: 'Product and amount required' });
  }

  const db = createDB(DB_NAME);
  const result = db.prepare(`
    INSERT INTO orders (user_id, product, amount, status) VALUES (?, ?, ?, 'pending')
  `).run(req.user.id, product, amount);

  res.status(201).json({
    message: 'Order created',
    data: { id: result.lastInsertRowid, product, amount, status: 'pending' }
  });
});

// ─── STATS endpoint (protected, admin only) ───────────────────────────────────
app.get('/stats', authMiddleware, (req, res) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Admin access required' });
  }

  const db = createDB(DB_NAME);
  const userCount = db.prepare(`SELECT COUNT(*) as count FROM users`).get();
  const orderCount = db.prepare(`SELECT COUNT(*) as count FROM orders`).get();
  const revenue = db.prepare(`SELECT SUM(amount) as total FROM orders WHERE status = 'completed'`).get();
  const statusBreakdown = db.prepare(`
    SELECT status, COUNT(*) as count FROM orders GROUP BY status
  `).all();

  res.json({
    users: userCount.count,
    orders: orderCount.count,
    revenue: revenue.total || 0,
    order_breakdown: statusBreakdown
  });
});

// ─── Start server ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 ACME Production App running on http://localhost:${PORT}`);
  console.log(`📦 Database: ${DB_NAME}`);
  console.log(`\nEndpoints:`);
  console.log(`  GET  /health         - Health check (public)`);
  console.log(`  POST /auth/login     - Login (public)`);
  console.log(`  GET  /users          - List users (auth required)`);
  console.log(`  GET  /users/:id      - Get user (auth required)`);
  console.log(`  GET  /orders         - List orders (auth required)`);
  console.log(`  POST /orders         - Create order (auth required)`);
  console.log(`  GET  /stats          - Admin stats (admin only)`);
});

module.exports = app;
