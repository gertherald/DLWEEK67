# 🛡️ Shadow Production Twin Demo

> *"Codex never touches production first — it must survive the shadow world."*

A safe, human-governed system that validates AI-proposed code changes before they ever reach production.

---

## 📋 Checklist Status

### 1. The "Production" App ✅
- [x] Express app with 5 endpoints (`/health`, `/auth/login`, `/users`, `/orders`, `/stats`)
- [x] SQLite database with full schema (users, orders, sessions)
- [x] Seed file with Faker.js data (50 users, 200 orders, 10 sessions)

### 2. Shadow Twin Environment ✅
- [x] Identical copy of app on port 3001
- [x] Its own seeded SQLite database (`shadow.sqlite`)
- [x] Docker Compose to spin both up

### 3. Test Suite ✅
- [x] Test 1: Schema integrity (tables not dropped/altered)
- [x] Test 2: Auth enforcement (all protected routes return 401 without token)
- [x] Test 3: Endpoint health (all endpoints return correct status codes)
- [x] Test 4: Data integrity (foreign keys, constraints intact)

### 4. GitHub Setup ⏳
- [ ] Create GitHub repo
- [ ] Push this code
- [ ] Set shadow tests as required status check in Branch Protection Rules

### 5. GitHub Actions Workflow ✅
- [x] Triggers on every PR to main/production
- [x] Spins up shadow environment
- [x] Runs full test suite
- [x] Posts result as PR comment

### 6. Demo PRs ✅
- [x] Bad PR: `demo/bad-change-no-auth.js` — removes auth from all routes
- [x] Bad PR: `demo/bad-change-drops-table.js` — drops users & sessions tables
- [ ] Good PR: Any small safe change (e.g., add a comment, update a log message)

### 7. Demo Video Flow ⏳
- [ ] Record bad PR getting blocked with logs
- [ ] Record good PR passing and merging
- [ ] Keep under 2 minutes

---

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Seed the production database + start the server
npm run dev

# In another terminal: seed shadow DB + run tests
DB_NAME=shadow.sqlite node src/seed.js
PORT=3001 DB_NAME=shadow.sqlite node src/app.js &
node tests/shadow-tests.js
```

## 🐳 Docker Start

```bash
docker-compose up
```

This starts:
- **Production** app on `http://localhost:3000`
- **Shadow Twin** on `http://localhost:3001`

---

## 🔑 API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | ❌ | Health check |
| POST | /auth/login | ❌ | Login with email |
| GET | /users | ✅ | List all users |
| GET | /users/:id | ✅ | Get single user |
| GET | /orders | ✅ | List all orders |
| POST | /orders | ✅ | Create order |
| GET | /stats | ✅ Admin | System stats |

---

## 🎬 Demo Video Script

**[0:00]** Show the production app running, explain what it is

**[0:20]** "Codex just proposed a change that removes authentication from all user routes"

**[0:30]** Open the bad PR in GitHub — show the code change

**[0:45]** Watch the GitHub Action trigger automatically

**[1:00]** Show the Shadow Twin test output — `❌ FAIL: /users returns 401 without token`

**[1:15]** Show the PR blocked with the comment: `🚨 Shadow Twin Decision: BLOCKED`

**[1:25]** "Now let's see a safe change pass"

**[1:35]** Open a good PR (just a comment change)

**[1:45]** Show all tests passing: `✅ Shadow Twin Decision: APPROVED`

**[2:00]** End

---

## 🏗️ Architecture

```
Codex (AI Developer)
        │
        ▼
   GitHub PR
        │
        ▼
GitHub Actions (triggers automatically)
        │
        ▼
Shadow Twin spins up (port 3001)
        │
   ┌────┴────┐
   │  Tests  │
   │ Schema  │
   │  Auth   │
   │ Health  │
   │  Data   │
   └────┬────┘
        │
   ┌────┴────┐
   │  PASS?  │
   └────┬────┘
  YES ──┤── NO
   │         │
   ▼         ▼
Approved   BLOCKED
 Merge     Human review
```

# trigger


<!-- harmless PR test -->

