/**
 * SHADOW TWIN TEST SUITE
 * 
 * These tests run against the Shadow environment BEFORE any change
 * is allowed to reach production. If ANY test fails, the PR is blocked.
 */

const http = require('http');
const { createDB } = require('../src/database');

const BASE_URL = `http://localhost:${process.env.PORT || 3001}`;
const SHADOW_DB = process.env.DB_NAME || 'shadow.sqlite';

let passed = 0;
let failed = 0;
const results = [];

// ─── Test Helper ──────────────────────────────────────────────────────────────
async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✅ PASS: ${name}`);
    results.push({ name, status: 'PASS' });
    passed++;
  } catch (err) {
    console.log(`  ❌ FAIL: ${name}`);
    console.log(`     └─ ${err.message}`);
    results.push({ name, status: 'FAIL', error: err.message });
    failed++;
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchJSON(path, options = {}) {
  const fetch = (await import('node-fetch')).default;
  const res = await fetch(`${BASE_URL}${path}`, options);
  return { status: res.status, body: await res.json() };
}

function getToken() {
  try {
    return require('fs').readFileSync('demo-token.txt', 'utf8').trim();
  } catch {
    // Fallback: get any valid token from shadow DB
    const db = createDB(SHADOW_DB);
    const session = db.prepare(`
      SELECT token FROM sessions WHERE expires_at > datetime('now') LIMIT 1
    `).get();
    return session?.token || '';
  }
}

// ─── TEST 1: Schema Integrity ─────────────────────────────────────────────────
async function testSchemaIntegrity() {
  console.log('\n📋 Test Group 1: Schema Integrity');

  await test('users table exists with correct columns', () => {
    const db = createDB(SHADOW_DB);
    const tableInfo = db.prepare(`PRAGMA table_info(users)`).all();
    const columns = tableInfo.map(c => c.name);

    assert(columns.includes('id'), 'Missing column: id');
    assert(columns.includes('name'), 'Missing column: name');
    assert(columns.includes('email'), 'Missing column: email');
    assert(columns.includes('role'), 'Missing column: role');
    assert(columns.includes('created_at'), 'Missing column: created_at');
  });

  await test('orders table exists with correct columns', () => {
    const db = createDB(SHADOW_DB);
    const tableInfo = db.prepare(`PRAGMA table_info(orders)`).all();
    const columns = tableInfo.map(c => c.name);

    assert(columns.includes('id'), 'Missing column: id');
    assert(columns.includes('user_id'), 'Missing column: user_id');
    assert(columns.includes('product'), 'Missing column: product');
    assert(columns.includes('amount'), 'Missing column: amount');
    assert(columns.includes('status'), 'Missing column: status');
  });

  await test('sessions table exists with correct columns', () => {
    const db = createDB(SHADOW_DB);
    const tableInfo = db.prepare(`PRAGMA table_info(sessions)`).all();
    const columns = tableInfo.map(c => c.name);

    assert(columns.includes('token'), 'Missing column: token — AUTH WILL BREAK');
    assert(columns.includes('expires_at'), 'Missing column: expires_at');
    assert(columns.includes('user_id'), 'Missing column: user_id');
  });

  await test('critical tables have not been dropped', () => {
    const db = createDB(SHADOW_DB);
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table'
    `).all().map(t => t.name);

    assert(tables.includes('users'), '🚨 CRITICAL: users table was DROPPED');
    assert(tables.includes('orders'), '🚨 CRITICAL: orders table was DROPPED');
    assert(tables.includes('sessions'), '🚨 CRITICAL: sessions table was DROPPED');
  });
}

// ─── TEST 2: Auth Enforcement ─────────────────────────────────────────────────
async function testAuthEnforcement() {
  console.log('\n🔐 Test Group 2: Auth Enforcement');

  await test('/users returns 401 without token', async () => {
    const { status } = await fetchJSON('/users');
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('/orders returns 401 without token', async () => {
    const { status } = await fetchJSON('/orders');
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('/stats returns 401 without token', async () => {
    const { status } = await fetchJSON('/stats');
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('invalid token is rejected', async () => {
    const { status } = await fetchJSON('/users', {
      headers: { 'Authorization': 'Bearer fake-token-12345' }
    });
    assert(status === 401, `Invalid token was accepted — SECURITY RISK`);
  });

  await test('valid token grants access to /users', async () => {
    const token = getToken();
    const { status } = await fetchJSON('/users', {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    assert(status === 200, `Valid token rejected — Expected 200, got ${status}`);
  });
}

// ─── TEST 3: Endpoint Health ──────────────────────────────────────────────────
async function testEndpointHealth() {
  console.log('\n🌐 Test Group 3: Endpoint Health');
  const token = getToken();
  const authHeaders = { 'Authorization': `Bearer ${token}` };

  await test('GET /health returns 200', async () => {
    const { status, body } = await fetchJSON('/health');
    assert(status === 200, `Health check failed: ${status}`);
    assert(body.status === 'ok', `Health check status not ok: ${body.status}`);
  });

  await test('GET /users returns data array', async () => {
    const { status, body } = await fetchJSON('/users', { headers: authHeaders });
    assert(status === 200, `Expected 200, got ${status}`);
    assert(Array.isArray(body.data), 'Response missing data array');
    assert(body.data.length > 0, 'Users table is empty — seeding may have failed');
  });

  await test('GET /orders returns data array', async () => {
    const { status, body } = await fetchJSON('/orders', { headers: authHeaders });
    assert(status === 200, `Expected 200, got ${status}`);
    assert(Array.isArray(body.data), 'Response missing data array');
  });

  await test('POST /orders creates a new order', async () => {
    const { status, body } = await fetchJSON('/orders', {
      method: 'POST',
      headers: { ...authHeaders, 'Content-Type': 'application/json' },
      body: JSON.stringify({ product: 'Shadow Test Plan', amount: 99.99 })
    });
    assert(status === 201, `Order creation failed: ${status}`);
    assert(body.data?.id, 'No order ID returned');
  });

  await test('GET /users/:id returns single user', async () => {
    const { status, body } = await fetchJSON('/users/1', { headers: authHeaders });
    assert(status === 200, `Expected 200, got ${status}`);
    assert(body.data?.id, 'No user data returned');
  });
}

// ─── TEST 4: Data Integrity ───────────────────────────────────────────────────
async function testDataIntegrity() {
  console.log('\n🗄️  Test Group 4: Data Integrity');

  await test('database has seed data (users exist)', () => {
    const db = createDB(SHADOW_DB);
    const count = db.prepare(`SELECT COUNT(*) as count FROM users`).get();
    assert(count.count > 0, 'No users found — database may have been wiped');
  });

  await test('foreign key relationships intact', () => {
    const db = createDB(SHADOW_DB);
    const orphanOrders = db.prepare(`
      SELECT COUNT(*) as count FROM orders o
      LEFT JOIN users u ON o.user_id = u.id
      WHERE u.id IS NULL
    `).get();
    assert(orphanOrders.count === 0, `${orphanOrders.count} orphan orders found — FK integrity broken`);
  });

  await test('email uniqueness constraint still enforced', () => {
    const db = createDB(SHADOW_DB);
    const duplicates = db.prepare(`
      SELECT email, COUNT(*) as count FROM users 
      GROUP BY email HAVING count > 1
    `).all();
    assert(duplicates.length === 0, `${duplicates.length} duplicate emails found`);
  });
}

// ─── Run All Tests ────────────────────────────────────────────────────────────
async function runAllTests() {
  console.log('\n');
  console.log('╔════════════════════════════════════════╗');
  console.log('║      SHADOW TWIN TEST SUITE            ║');
  console.log('║   Testing proposed change safety...    ║');
  console.log('╚════════════════════════════════════════╝');

  // Wait for server to be ready
  await new Promise(resolve => setTimeout(resolve, 1000));

  await testSchemaIntegrity();
  await testAuthEnforcement();
  await testEndpointHealth();
  await testDataIntegrity();

  // ─── Summary ─────────────────────────────────────────────────────────────
  console.log('\n');
  console.log('╔════════════════════════════════════════╗');
  console.log(`║  Results: ${passed} passed, ${failed} failed          ║`);
  console.log('╚════════════════════════════════════════╝');

  if (failed > 0) {
    console.log('\n🚨 SHADOW TWIN DECISION: BLOCKED');
    console.log('   This change has been rejected. Human review required.\n');
    process.exit(1); // Exit code 1 = GitHub Actions marks PR as failed
  } else {
    console.log('\n✅ SHADOW TWIN DECISION: APPROVED');
    console.log('   All checks passed. Change is safe to merge.\n');
    process.exit(0); // Exit code 0 = GitHub Actions marks PR as passed
  }
}

runAllTests().catch(err => {
  console.error('Test runner crashed:', err);
  process.exit(1);
});
