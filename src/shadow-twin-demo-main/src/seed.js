const { createDB } = require('./database');
const crypto = require('crypto');
const fs = require('fs');

const DB_NAME = process.env.DB_NAME || 'production.sqlite';

console.log(`\n🌱 Seeding database: ${DB_NAME}`);
const db = createDB(DB_NAME);

// Clear existing data
db.exec(`DELETE FROM sessions; DELETE FROM orders; DELETE FROM users;`);

// Helper functions to generate fake data without faker
function randomName() {
  const first = ['James','Sarah','Michael','Emma','John','Olivia','David','Sophia','Chris','Ava'];
  const last = ['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis','Wilson','Moore'];
  return `${first[Math.floor(Math.random()*first.length)]} ${last[Math.floor(Math.random()*last.length)]}`;
}

function randomEmail(name) {
  const domains = ['gmail.com','yahoo.com','outlook.com','company.com','work.io'];
  const slug = name.toLowerCase().replace(' ', '.') + Math.floor(Math.random()*999);
  return `${slug}@${domains[Math.floor(Math.random()*domains.length)]}`;
}

// Seed Users (50 fake users)
const insertUser = db.prepare(`INSERT INTO users (name, email, role) VALUES (?, ?, ?)`);

const users = [];
for (let i = 0; i < 50; i++) {
  const name = randomName();
  const email = randomEmail(name) + i; // ensure uniqueness
  const role = i === 0 ? 'admin' : 'user';
  const result = insertUser.run(name, email, role);
  users.push({ id: result.lastInsertRowid, name, email });
}
console.log(`✅ Created ${users.length} users`);

// Seed Orders (200 fake orders)
const insertOrder = db.prepare(`INSERT INTO orders (user_id, product, amount, status) VALUES (?, ?, ?, ?)`);
const products = ['Premium Plan', 'Basic Plan', 'Enterprise Plan', 'Add-on Pack', 'Support Tier'];
const statuses = ['pending', 'completed', 'cancelled', 'refunded'];

for (let i = 0; i < 200; i++) {
  const user = users[Math.floor(Math.random() * users.length)];
  const product = products[Math.floor(Math.random() * products.length)];
  const amount = parseFloat((Math.random() * 990 + 9).toFixed(2));
  const status = statuses[Math.floor(Math.random() * statuses.length)];
  insertOrder.run(user.id, product, amount, status);
}
console.log(`✅ Created 200 orders`);

// Seed Sessions
const insertSession = db.prepare(`INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)`);
let demoToken = '';

for (let i = 0; i < 10; i++) {
  const token = crypto.randomBytes(32).toString('hex');
  const expiresAt = new Date(Date.now() + 86400000).toISOString();
  insertSession.run(users[i].id, token, expiresAt);
  if (i === 0) demoToken = token;
}

fs.writeFileSync('demo-token.txt', demoToken);
console.log(`✅ Created 10 sessions`);
console.log(`\n🎉 Seeding complete!`);
console.log(`🔑 Demo token: ${demoToken}\n`);
