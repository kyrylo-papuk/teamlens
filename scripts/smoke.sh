#!/bin/sh
# End-to-end check of the running compose stack. No model key is needed: it exercises what is plain code
# and the path between the two services.  Usage:  sh scripts/smoke.sh [base url]
set -eu
BASE="${1:-http://localhost:8000}"

PYTHON="${PYTHON:-python3}"

json() { "$PYTHON" -c "import sys, json; d = json.load(sys.stdin); print($1)" | tr -d '\r'; }
post() { curl -fsS -X POST "$BASE$1" -H "Content-Type: application/json" -d "$2"; }
expect() { [ "$1" = "$2" ] || { echo "FAIL: $3: expected '$2', got '$1'"; exit 1; }; echo "ok   $3"; }

expect "$(curl -fsS "$BASE/api/people" | json "len([p for p in d['people'] if p['team'] == 'PAY']), [p['level'] for p in d['people']].count('executive')")" \
       "8 3" "teams and leadership come from the People agent over A2A"

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

KEY2="$(echo "$OVERVIEW" | json "d['backlog'][0]")"
expect "$(post /api/issues/assign "{\"actor\": \"E004\", \"issue_key\": \"$KEY2\", \"assignee_id\": \"E004\"}" | json "d['ok']")" "True" "an engineer takes unowned work with a button"
expect "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/api/issues/move" -H 'Content-Type: application/json' \
          -d "{\"actor\": \"E005\", \"issue_key\": \"$KEY2\", \"status\": \"In Review\"}")" "403" "a colleague cannot move it"

expect "$(curl -fsS "$BASE/api/company?actor=E149" | json "len(d['teams'])")" "5" "leadership sees every team"
expect "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/company?actor=E001")" "403" "a manager's view ends at the team"
expect "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/audit?actor=E003")" "403" "an engineer cannot read the audit log"

expect "$(curl -fsS "$BASE/api/audit?actor=E001" | json "d['intact'], d['total'] >= 5")" "True True" "everything above is in an intact audit chain"
echo "all checks passed"
