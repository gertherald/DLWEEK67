/**
 * shadow-ttt-tests.js — Shadow Twin Test Suite for TicTacToe Game Server
 *
 * Runs against the shadow TTT server at localhost:PORT.
 * Uses Node.js built-in fetch (Node 18+) — no node-fetch dependency.
 *
 * Exit 0 → all tests pass → SHADOW TWIN: APPROVED
 * Exit 1 → any test fails → SHADOW TWIN: BLOCKED
 *
 * Test groups:
 *   1. Auth enforcement  — protected routes return 401 without a valid token
 *   2. Input validation  — player names with XSS/SQL payloads are rejected (400)
 *   3. Game logic        — start game, make moves, detect occupied cells
 *   4. Schema integrity  — games table exists with required columns
 */

const fs = require('fs');
const { createDB } = require('../src/database');

const BASE_URL  = `http://localhost:${process.env.PORT || 3001}`;
const DB_NAME   = process.env.DB_NAME || 'shadow.sqlite';

let passed = 0;
let failed = 0;

// ─── Test helpers ──────────────────────────────────────────────────────────────
async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✅ PASS: ${name}`);
    passed++;
  } catch (err) {
    console.log(`  ❌ FAIL: ${name}`);
    console.log(`     └─ ${err.message}`);
    failed++;
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function req(path, options = {}) {
  const res  = await fetch(`${BASE_URL}${path}`, options);
  let body;
  try { body = await res.json(); } catch { body = {}; }
  return { status: res.status, body };
}

// ─── Get auth token from demo-token.txt or shadow DB directly ─────────────────
function getToken() {
  try {
    const token = fs.readFileSync('demo-token.txt', 'utf8').trim();
    if (token) return token;
  } catch { /* fall through */ }
  const db      = createDB(DB_NAME);
  const session = db.prepare(
    `SELECT token FROM sessions WHERE expires_at > datetime('now') LIMIT 1`
  ).get();
  return session?.token || '';
}

// ─── TEST GROUP 1: Auth Enforcement ───────────────────────────────────────────
async function testAuthEnforcement() {
  console.log('\n🔐 Test Group 1: Auth Enforcement');

  await test('/game/start returns 401 without token', async () => {
    const { status } = await req('/game/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_name: 'Alice' }),
    });
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('/leaderboard returns 401 without token', async () => {
    const { status } = await req('/leaderboard');
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('/game/state returns 401 without token', async () => {
    const { status } = await req('/game/1/state');
    assert(status === 401, `Expected 401 but got ${status} — AUTH IS BROKEN`);
  });

  await test('invalid token is rejected (401)', async () => {
    const { status } = await req('/game/start', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer totally-fake-token-000',
                 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_name: 'Alice' }),
    });
    assert(status === 401, `Fake token was accepted — Expected 401 got ${status}`);
  });

  await test('valid token grants access to /game/start (201)', async () => {
    const token = getToken();
    const { status } = await req('/game/start', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}`,
                 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_name: 'Alice' }),
    });
    assert(status === 201, `Valid token rejected — Expected 201 got ${status}`);
  });
}

// ─── TEST GROUP 2: Input Validation ───────────────────────────────────────────
async function testInputValidation() {
  console.log('\n🛡️  Test Group 2: Input Validation');
  const token = getToken();
  const authHeaders = { 'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json' };

  await test('XSS payload in player_name is rejected (400)', async () => {
    const { status, body } = await req('/game/start', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ player_name: '<script>alert(1)</script>' }),
    });
    assert(status === 400,
      `XSS payload was accepted (status ${status}) — INPUT VALIDATION BROKEN`);
  });

  await test('SQL injection in player_name is rejected (400)', async () => {
    const { status } = await req('/game/start', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ player_name: "'; DROP TABLE games;--" }),
    });
    assert(status === 400,
      `SQL injection accepted (status ${status}) — INPUT VALIDATION BROKEN`);
  });

  await test('position out of range (-1) is rejected (400)', async () => {
    // Start a game first so we have a valid game ID
    const start = await req('/game/start', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ player_name: 'ValidPlayer' }),
    });
    const game_id = start.body.game_id;
    const { status } = await req(`/game/${game_id}/move`, {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ position: -1 }),
    });
    assert(status === 400, `Position -1 was accepted (status ${status}) — VALIDATION BROKEN`);
  });

  await test('position out of range (9) is rejected (400)', async () => {
    const start = await req('/game/start', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ player_name: 'ValidPlayer2' }),
    });
    const game_id = start.body.game_id;
    const { status } = await req(`/game/${game_id}/move`, {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ position: 9 }),
    });
    assert(status === 400, `Position 9 was accepted (status ${status}) — VALIDATION BROKEN`);
  });
}

// ─── TEST GROUP 3: Game Logic ──────────────────────────────────────────────────
async function testGameLogic() {
  console.log('\n🎮 Test Group 3: Game Logic');
  const token = getToken();
  const authHeaders = { 'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json' };

  let game_id;

  await test('can start a new game (201)', async () => {
    const { status, body } = await req('/game/start', {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ player_name: 'TestPlayer' }),
    });
    assert(status === 201, `Expected 201 got ${status}`);
    assert(body.game_id,   'No game_id in response');
    assert(body.board === '---------', `Unexpected board: ${body.board}`);
    game_id = body.game_id;
  });

  await test('can retrieve game state', async () => {
    if (!game_id) return;
    const { status, body } = await req(`/game/${game_id}/state`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    assert(status === 200, `Expected 200 got ${status}`);
    assert(body.data?.id,  'No game data in response');
  });

  await test('can make a valid move (position 4)', async () => {
    if (!game_id) return;
    const { status, body } = await req(`/game/${game_id}/move`, {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ position: 4 }),
    });
    assert(status === 200, `Expected 200 got ${status}`);
    assert(body.board[4] === 'X', `Expected X at position 4, got: ${body.board}`);
  });

  await test('cannot play on an occupied cell (400)', async () => {
    if (!game_id) return;
    const { status } = await req(`/game/${game_id}/move`, {
      method: 'POST',
      headers: authHeaders,
      body: JSON.stringify({ position: 4 }),   // cell 4 already played
    });
    assert(status === 400, `Occupied cell was accepted (status ${status})`);
  });

  await test('/health returns status ok', async () => {
    const { status, body } = await req('/health');
    assert(status === 200, `Health check failed: ${status}`);
    assert(body.status === 'ok', `Health status not ok: ${body.status}`);
  });
}

// ─── TEST GROUP 4: Schema Integrity ───────────────────────────────────────────
async function testSchemaIntegrity() {
  console.log('\n📋 Test Group 4: Schema Integrity');

  await test('games table exists in shadow DB', () => {
    const db     = createDB(DB_NAME);
    const tables = db.prepare(
      `SELECT name FROM sqlite_master WHERE type='table'`
    ).all().map(t => t.name);
    assert(tables.includes('games'),
      '🚨 games table missing — TTT server may not have been started');
  });

  await test('games table has required columns', () => {
    const db      = createDB(DB_NAME);
    const cols    = db.prepare(`PRAGMA table_info(games)`).all().map(c => c.name);
    const required = ['id', 'player_x', 'board', 'current_player', 'status', 'winner'];
    for (const col of required) {
      assert(cols.includes(col), `Missing column in games table: ${col}`);
    }
  });

  await test('sessions table still intact (auth not broken at DB level)', () => {
    const db   = createDB(DB_NAME);
    const info = db.prepare(`PRAGMA table_info(sessions)`).all().map(c => c.name);
    assert(info.includes('token'),
      '🚨 sessions.token column missing — auth will be broken');
  });

  await test('seed data present (users exist)', () => {
    const db    = createDB(DB_NAME);
    const count = db.prepare(`SELECT COUNT(*) as n FROM users`).get();
    assert(count.n > 0, 'No users found — seed.js may not have run');
  });
}

// ─── Run all tests ─────────────────────────────────────────────────────────────
async function runAllTests() {
  console.log('\n');
  console.log('╔══════════════════════════════════════════════╗');
  console.log('║    SHADOW TWIN — TicTacToe Test Suite        ║');
  console.log('║    Testing proposed change safety ...        ║');
  console.log('╚══════════════════════════════════════════════╝');

  // Brief pause to let the server finish starting
  await new Promise(r => setTimeout(r, 800));

  await testAuthEnforcement();
  await testInputValidation();
  await testGameLogic();
  await testSchemaIntegrity();

  console.log('\n');
  console.log('╔══════════════════════════════════════════════╗');
  console.log(`║  Results: ${String(passed).padEnd(2)} passed,  ${String(failed).padEnd(2)} failed              ║`);
  console.log('╚══════════════════════════════════════════════╝');

  if (failed > 0) {
    console.log('\n🚨 SHADOW TWIN DECISION: BLOCKED');
    console.log('   This change has been rejected. Human review required.\n');
    process.exit(1);
  } else {
    console.log('\n✅ SHADOW TWIN DECISION: APPROVED');
    console.log('   All checks passed. Change is safe to merge.\n');
    process.exit(0);
  }
}

runAllTests().catch(err => {
  console.error('Test runner crashed:', err.message);
  process.exit(1);
});
