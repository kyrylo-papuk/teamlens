#!/bin/sh
# End-to-end check of the running compose stack. No model key is needed: it exercises what is plain code
# and the path between the two services.  Usage:  sh scripts/smoke.sh [base url]
set -eu
BASE="${1:-http://localhost:8000}"

PYTHON="${PYTHON:-python3}"

json() { "$PYTHON" -c "import sys, json; d = json.load(sys.stdin); print($1)" | tr -d '\r'; }
post() { curl -fsS -X POST "$BASE$1" -H "Content-Type: application/json" -d "$2"; }
expect() { [ "$1" = "$2" ] || { echo "FAIL: $3: expected '$2', got '$1'"; exit 1; }; echo "ok   $3"; }

expect "$(curl -fsS "$BASE/api/people" | json "d['team'], len(d['people'])")" "PAY 8" "the team comes from the People agent over A2A"

OVERVIEW="$(curl -fsS "$BASE/api/overview?actor=E001")"
expect "$(echo "$OVERVIEW" | json "[f['kind'] for f in d['findings']]")" "['assigned_while_absent']" "the planted problem is found"
expect "$(echo "$OVERVIEW" | json "'sick' in json.dumps(d) or 'salary' in json.dumps(d)")" "False" "no restricted HR data in the overview"
expect "$(echo "$OVERVIEW" | json "d['forecast']['committed'] > 0")" "True" "the sprint forecast is there"

KEY="$(post /api/issues '{"actor": "E001", "summary": "Smoke test task", "points": 1}' | json "d['issue_key']")"
expect "$(curl -fsS "$BASE/api/autoassign?actor=E001" | json "d['suggestions'][0]['issue_key'] == '$KEY' and bool(d['suggestions'][0]['assignee_id'])")" \
       "True" "a new sprint task gets a suggested owner"
expect "$(post /api/autoassign/assign "{\"actor\": \"E001\", \"issue_key\": \"$KEY\"}" | json "d['ok']")" "True" "the suggestion can be confirmed"
expect "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/autoassign/mode" -H 'Content-Type: application/json' \
          -d '{"actor": "E003", "mode": "auto"}')" "403" "an engineer cannot switch automatic assignment"

# Leftovers of an earlier run would make the next two calls fail; a fresh stack has none.
post /api/roles/delete '{"actor": "E001", "name": "tech-lead"}' > /dev/null 2>&1 || true
post /api/roles/define '{"actor": "E001", "name": "tech-lead", "rights": ["issues.assign"]}' > /dev/null
post /api/roles/grant '{"actor": "E001", "employee_id": "E003", "role": "tech-lead"}' > /dev/null
expect "$(curl -fsS "$BASE/api/roles?actor=E003" | json "'issues.assign' in d['mine']")" "True" "a role given by the manager takes effect"

expect "$(curl -fsS "$BASE/api/audit" | json "d['intact'], d['total'] >= 4")" "True True" "everything above is in an intact audit chain"
echo "all checks passed"
