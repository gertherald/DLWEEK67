const { createDB } = require('./database');

const DB_NAME = process.env.DB_NAME || 'production.sqlite';

function authMiddleware(req, res, next) {
  const token = req.headers['authorization']?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Missing authorization token'
    });
  }

  const db = createDB(DB_NAME);
  const session = db.prepare(`
    SELECT s.*, u.name, u.email, u.role 
    FROM sessions s 
    JOIN users u ON s.user_id = u.id 
    WHERE s.token = ? AND s.expires_at > datetime('now')
  `).get(token);

  if (!session) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid or expired token'
    });
  }

  req.user = {
    id: session.user_id,
    name: session.name,
    email: session.email,
    role: session.role
  };

  next();
}

module.exports = { authMiddleware };
