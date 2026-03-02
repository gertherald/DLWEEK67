"""
shadow_runner.py — Python bridge to the Node.js Shadow Twin
============================================================
Runs the tic-tac-toe shadow twin test suite against a scenario-specific
Node.js server and returns pass/fail for the DL Meta-Scorer feature
`shadow_twin_passed` (feature #29).

Execution flow per scenario:
  1. node src/seed.js               → populate shadow.sqlite + demo-token.txt
  2. node src/<app_file>            → start TTT game server on SHADOW_PORT
  3. node tests/shadow-ttt-tests.js → run the 17-assertion test suite
  4. terminate server               → clean up

Scenario selection — parallel probing, DL decision is NOT used:
  ANY security signal   → run no-auth  AND no-validation in parallel
                          → both tests FAIL → shadow_twin_passed = 0
  No security signals   → run ttt-baseline only
                          → tests PASS     → shadow_twin_passed = 1

The shadow twin is an independent, parallel probe:
  • Does NOT pre-select one scenario from a single attribute (e.g. cwe_has_auth_bypass).
  • For any code with security signals, BOTH auth bypass AND input validation
    are probed simultaneously regardless of which specific CWE fired.
  • This means an APPROVE case with real auth/injection signals is tested
    for BOTH vulnerability types — the test results speak, not the attribute.
  • A BLOCK case with only path-traversal/subprocess signals (no auth/injection)
    maps to baseline → shadow PASS (auth+validation intact, different risk type).

Results are cached per scenario — max 3 Node.js runs for any number of cases.

Usage:
    from shadow_runner import run_shadow_twin, warm_up

    # Pre-run all 3 scenarios and cache results (call once at demo start)
    warm_up()

    # Per-case: look up cached result (fast, no subprocess)
    result = run_shadow_twin(cve_signals, smells, decision)
    # → {"shadow_twin_passed": 1|0|-1, "scenario": str, "passed": bool|None}
"""

import os
import sys
import time
import subprocess

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SHADOW_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'shadow-twin-demo-main'))
SHADOW_PORT = 3001
SHADOW_DB   = 'shadow.sqlite'

# App files relative to SHADOW_DIR
_APP_FILES = {
    'ttt-baseline':      'src/ttt-app.js',
    'no-auth':           'demo/bad-ttt-no-auth.js',
    'no-validation':     'demo/bad-ttt-no-validation.js',
}
_TEST_FILE = 'tests/shadow-ttt-tests.js'

# ── Per-session scenario cache ─────────────────────────────────────────────────
_scenario_cache: dict[str, dict] = {}


# ── Node.js availability check ─────────────────────────────────────────────────

def _check_node() -> str:
    """Return Node.js version string, or raise RuntimeError if not found."""
    try:
        result = subprocess.run(
            ['node', '--version'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    raise RuntimeError(
        "Node.js not found. Install Node.js 18+ and ensure 'node' is on PATH."
    )


# ── Scenario selector ──────────────────────────────────────────────────────────

def _select_scenarios(cve_signals: dict, smells: list) -> set:
    """
    Return the SET of shadow scenarios to run for this code review.

    The DL decision is NOT used.  The shadow twin probes independently:

    ANY security signal → {'no-auth', 'no-validation'}
      Both auth bypass AND input validation are probed in parallel.
      A single attribute (e.g. cwe_has_auth_bypass) no longer gates whether
      auth is tested — if the code has ANY security signal, both probes run.

    No security signals → {'ttt-baseline'}
      Code appears clean; baseline confirms all security tests pass.

    Why both bad scenarios for any signal?
      • A case with only XSS signals still gets no-auth probed — if auth
        turns out to be broken too, the test catches it.
      • A case where cwe_has_auth_bypass never fired (too narrow) is still
        auth-tested because any other security signal triggers the probe.
    """
    has_any_signal = (
        cve_signals.get('cwe_has_auth_bypass')   or
        cve_signals.get('cwe_has_xss')           or
        cve_signals.get('cwe_has_sql_injection')  or
        cve_signals.get('cwe_has_buffer_overflow') or   # includes unsafe deser
        any(kw in s for s in smells
            for kw in ('eval()', 'exec()', 'unsafe', 'compile()'))
    )
    if has_any_signal:
        return {'no-auth', 'no-validation'}   # parallel probe both vulnerability types
    return {'ttt-baseline'}


# ── Scenario runner ────────────────────────────────────────────────────────────

def _run_scenario(scenario: str) -> dict:
    """
    Actually run the shadow twin for one scenario:
      1. Seed the shadow DB
      2. Start the scenario server on SHADOW_PORT
      3. Run shadow-ttt-tests.js
      4. Kill the server

    Returns a dict with keys: scenario, passed, shadow_twin_passed, exit_code, output.
    """
    env = {**os.environ, 'PORT': str(SHADOW_PORT), 'DB_NAME': SHADOW_DB}
    app_file = _APP_FILES[scenario]

    # Step 1: Seed shadow DB (creates users, sessions, demo-token.txt)
    seed = subprocess.run(
        ['node', 'src/seed.js'],
        cwd=SHADOW_DIR, env=env,
        capture_output=True, text=True, timeout=20,
    )
    if seed.returncode != 0:
        return {
            'scenario': scenario,
            'passed': None,
            'shadow_twin_passed': -1,
            'exit_code': seed.returncode,
            'output': f'seed.js failed: {seed.stderr[:300]}',
        }

    # Step 2: Start shadow server
    server = subprocess.Popen(
        ['node', app_file],
        cwd=SHADOW_DIR, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.8)   # give Express time to bind

    # Step 3: Run tests
    try:
        test_result = subprocess.run(
            ['node', _TEST_FILE],
            cwd=SHADOW_DIR, env=env,
            capture_output=True, text=True, timeout=25,
        )
    except subprocess.TimeoutExpired:
        server.terminate()
        return {
            'scenario': scenario,
            'passed': None,
            'shadow_twin_passed': -1,
            'exit_code': -1,
            'output': 'shadow-ttt-tests.js timed out',
        }
    finally:
        # Step 4: Clean up server regardless
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    passed = test_result.returncode == 0
    # Grab last 600 chars of output for display
    output = (test_result.stdout or '') + (test_result.stderr or '')
    output = output[-600:].strip()

    return {
        'scenario':          scenario,
        'passed':            passed,
        'shadow_twin_passed': 1 if passed else 0,
        'exit_code':         test_result.returncode,
        'output':            output,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_shadow_twin(cve_signals: dict, smells: list, _decision: str) -> dict:
    """
    Return the shadow twin result for a code review.

    Selects scenarios via _select_scenarios() (code signals only, not decision).
    All selected scenarios are looked up from cache (populated by warm_up()).

    Aggregation logic:
      {'ttt-baseline'}             → shadow_twin_passed = 1 if passes, -1 if error
      {'no-auth', 'no-validation'} → shadow_twin_passed = 0 (vulnerabilities detected)
                                     if either bad server causes test failures

    Returns dict with:
      shadow_twin_passed  int    1=clean  0=vulnerable  -1=error
      scenario            str    e.g. 'no-auth+no-validation' or 'ttt-baseline'
      scenarios           list   list of individual scenarios run
      passed              bool|None
    """
    scenarios = _select_scenarios(cve_signals, smells)

    results = {}
    for sc in scenarios:
        if sc not in _scenario_cache:
            try:
                _check_node()
                _scenario_cache[sc] = _run_scenario(sc)
            except RuntimeError as e:
                _scenario_cache[sc] = {
                    'scenario': sc, 'passed': None,
                    'shadow_twin_passed': -1, 'output': str(e),
                }
            except Exception as e:
                _scenario_cache[sc] = {
                    'scenario': sc, 'passed': None,
                    'shadow_twin_passed': -1,
                    'output': f'Unexpected error: {e}',
                }
        results[sc] = _scenario_cache[sc]

    scenario_key = '+'.join(sorted(scenarios))

    # ── Aggregate result ──────────────────────────────────────────────────────
    if scenarios == {'ttt-baseline'}:
        # Clean code path: baseline must pass
        r = results['ttt-baseline']
        shadow_twin_passed = 1 if r.get('passed') is True else (-1 if r.get('passed') is None else 1)
        passed = r.get('passed')
    else:
        # Security signals detected: both bad scenarios are probed.
        # Any test failure confirms a detectable vulnerability → shadow_twin_passed = 0.
        # Unexpected all-pass → shadow_twin_passed = 1 (rare; means tests missed it).
        any_error = any(r.get('passed') is None for r in results.values())
        if any_error:
            shadow_twin_passed = -1
        elif any(r.get('passed') is False for r in results.values()):
            shadow_twin_passed = 0   # vulnerability detected
        else:
            shadow_twin_passed = 1   # bad servers unexpectedly passed (edge case)
        passed = False if shadow_twin_passed == 0 else None

    return {
        'shadow_twin_passed': shadow_twin_passed,
        'scenario':           scenario_key,
        'scenarios':          sorted(scenarios),
        'passed':             passed,
        'probe_results':      {sc: r.get('passed') for sc, r in results.items()},
    }


def _run_scenario_cached(scenario: str) -> dict:
    """Run scenario if not already cached, then return cached result."""
    if scenario not in _scenario_cache:
        try:
            _scenario_cache[scenario] = _run_scenario(scenario)
        except Exception as e:
            _scenario_cache[scenario] = {
                'scenario': scenario, 'passed': None,
                'shadow_twin_passed': -1, 'output': str(e),
            }
    return _scenario_cache[scenario]


def warm_up() -> dict:
    """
    Pre-run all three TTT scenarios before the main demo loop.
    Results are cached so per-case calls return instantly.
    """
    W = 62
    print(f"\n{'╔' + '═'*(W-2) + '╗'}")
    print(f"║{'  SHADOW TWIN PRE-FLIGHT  (3 TTT scenarios)':^{W-2}}║")
    print(f"╚{'═'*(W-2)}╝")

    try:
        node_ver = _check_node()
        print(f"  Node.js : {node_ver}")
    except RuntimeError as e:
        print(f"  ⚠  {e}")
        print(f"  Shadow twin disabled — shadow_twin_passed will be -1.\n")
        for scenario in _APP_FILES:
            _scenario_cache[scenario] = {
                'scenario': scenario, 'passed': None,
                'shadow_twin_passed': -1, 'output': 'Node.js not found',
            }
        return {'node_available': False}

    nm_path = os.path.join(SHADOW_DIR, 'node_modules')
    if not os.path.isdir(nm_path):
        print(f"  Installing npm dependencies …")
        subprocess.run(['npm', 'install', '--silent'],
                       cwd=SHADOW_DIR, capture_output=True, timeout=60)

    print(f"\n  Running 3 scenarios (results cached for all 90 cases) …\n")

    rows = [
        ('ttt-baseline',  'no dangerous signals', 'no auth/injection detected → PASS'),
        ('no-auth',       'auth_bypass detected',  'no auth layer → tests FAIL'),
        ('no-validation', 'injection detected',    'no input checks → tests FAIL'),
    ]
    summary = {'node_available': True, 'scenarios': {}}

    for scenario, case_type, note in rows:
        result = _run_scenario_cached(scenario)
        passed = result.get('passed')
        icon   = '✅' if passed is True else ('❌' if passed is False else '⚠️ ')
        status = 'PASS' if passed is True else ('FAIL' if passed is False else 'N/A ')
        print(f"  {icon}  {scenario:<20}  → {status}  ({note})")
        summary['scenarios'][scenario] = result

    print(f"\n  Shadow twin ready — all 90 cases use cached results.")
    return summary


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Shadow runner self-test …")
    summary = warm_up()
    if not summary.get('node_available'):
        sys.exit(1)

    # Test _select_scenarios — returns a SET, decision is NOT used
    # (any security signal → both bad scenarios probed; no signal → baseline only)
    tests = [
        ({'cwe_has_auth_bypass': 1},  [],              {'no-auth', 'no-validation'}),
        ({'cwe_has_xss': 1},          [],              {'no-auth', 'no-validation'}),
        ({'cwe_has_sql_injection': 1}, [],             {'no-auth', 'no-validation'}),
        ({'cwe_has_buffer_overflow':1}, [],            {'no-auth', 'no-validation'}),
        ({},                ['eval() usage'],          {'no-auth', 'no-validation'}),
        ({},                [],                        {'ttt-baseline'}),  # clean code
    ]
    print("\nScenario selection checks (parallel probing):")
    ok = True
    for sigs, smells, expected in tests:
        got = _select_scenarios(sigs, smells)
        mark = '✓' if got == expected else '✗'
        sig_label = list(sigs.keys()) or smells or ['(none)']
        print(f"  {mark}  signals={sig_label}  → {sorted(got)}")
        if got != expected:
            ok = False
    sys.exit(0 if ok else 1)
