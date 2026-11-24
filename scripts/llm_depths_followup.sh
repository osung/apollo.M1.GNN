#!/usr/bin/env bash
# Unattended follow-up of the depth-tiered LLM scoring (candidates.llm_depths):
# waits for the local and the remote scorer, restarts one that failed (scoring resumes
# from its score file), fetches the remote scores and evaluates every depth.
#
#   setsid nohup scripts/llm_depths_followup.sh > logs/llm_depths_followup.log 2>&1 < /dev/null &
#
# LOCAL_DIRECTION / REMOTE_DIRECTION: the direction each machine is scoring (one shard each).
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-192.168.100.2}
REMOTE_DIR=${REMOTE_DIR:-work/apollo.M1.GNN}
LOCAL_DIRECTION=${LOCAL_DIRECTION:-c2p}
REMOTE_DIRECTION=${REMOTE_DIRECTION:-p2c}
MAX_RESTARTS=${MAX_RESTARTS:-3}
POLL=${POLL:-300}
PY=.venv/bin/python
OUT=$($PY -c "import yaml; print(yaml.safe_load(open('config/llm_match.yaml'))['output_dir'])")
DEPTHS=$($PY -c "
import yaml
c = yaml.safe_load(open('config/llm_match.yaml'))['candidates']
print(*sorted({int(c['topn']), *map(int, c.get('llm_depths') or [])}))")
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE")

say() { echo "$(date '+[%m-%d %H:%M:%S]') $*"; }
local_log=logs/llm_score_depths_${LOCAL_DIRECTION}.log
remote_log=logs/llm_score_depths_${REMOTE_DIRECTION}.log
score_cmd() {  # $1 = direction, $2 = log file
  echo "make llm-score DIRECTION=$1 SHARD=0 NUM_SHARDS=1 >> $2 2>&1; echo \"exit \$?\" >> $2"
}

# last "exit N" line of a log ("" while the scorer is still running)
local_exit() { grep '^exit ' "$local_log" 2>/dev/null | tail -1 | cut -d' ' -f2; }
remote_exit() {
  "${SSH[@]}" "grep '^exit ' $REMOTE_DIR/$remote_log 2>/dev/null | tail -1 | cut -d' ' -f2" \
    < /dev/null 2> /dev/null
}
local_running() { pgrep -f "llm_match_scor[e].py --direction $LOCAL_DIRECTION" > /dev/null; }
remote_running() {
  "${SSH[@]}" "pgrep -f 'llm_match_scor[e].py --direction $REMOTE_DIRECTION'" \
    < /dev/null > /dev/null 2>&1
}

evaluate() {  # $1 = direction
  for depth in $DEPTHS; do
    say "evaluating $1 at depth $depth"
    $PY scripts/llm_match_eval.py --direction "$1" --topn "$depth" 2>&1 \
      | grep -v 'Warning\|warnings.warn' > "logs/llm_eval_${1}_top${depth}.log" \
      || say "! evaluation of $1 at depth $depth failed (see logs/llm_eval_${1}_top${depth}.log)"
  done
}

local_done=0 remote_done=0 local_restarts=0 remote_restarts=0
while [ "$local_done" = 0 ] || [ "$remote_done" = 0 ]; do
  if [ "$local_done" = 0 ] && ! local_running; then
    code=$(local_exit)
    if [ "$code" = 0 ]; then
      say "local $LOCAL_DIRECTION scoring finished"
      evaluate "$LOCAL_DIRECTION"
      local_done=1
    elif [ "$local_restarts" -lt "$MAX_RESTARTS" ]; then
      local_restarts=$((local_restarts + 1))
      say "local scorer is not running (exit '${code}'); restart $local_restarts/$MAX_RESTARTS"
      sed -i 's/^exit /exited /' "$local_log" 2> /dev/null
      setsid nohup bash -c "$(score_cmd "$LOCAL_DIRECTION" "$local_log")" > /dev/null 2>&1 < /dev/null &
      sleep 60
    else
      say "! local scorer failed $MAX_RESTARTS times; giving up on $LOCAL_DIRECTION"
      local_done=2
    fi
  fi
  if [ "$remote_done" = 0 ] && "${SSH[@]}" true < /dev/null 2> /dev/null && ! remote_running; then
    code=$(remote_exit)
    if [ "$code" = 0 ]; then
      say "remote $REMOTE_DIRECTION scoring finished; fetching scores"
      if make sync-results > /dev/null 2>&1; then
        evaluate "$REMOTE_DIRECTION"
        remote_done=1
      else
        say "! could not fetch the remote scores; will retry"
      fi
    elif [ "$remote_restarts" -lt "$MAX_RESTARTS" ]; then
      remote_restarts=$((remote_restarts + 1))
      say "remote scorer is not running (exit '${code}'); restart $remote_restarts/$MAX_RESTARTS"
      "${SSH[@]}" -f "cd $REMOTE_DIR && sed -i 's/^exit /exited /' $remote_log 2>/dev/null; \
        nohup bash -c '$(score_cmd "$REMOTE_DIRECTION" "$remote_log")' > /dev/null 2>&1" < /dev/null
      sleep 60
    else
      say "! remote scorer failed $MAX_RESTARTS times; giving up on $REMOTE_DIRECTION"
      remote_done=2
    fi
  fi
  [ "$local_done" != 0 ] && [ "$remote_done" != 0 ] && break
  sleep "$POLL"
done

$PY - "$OUT" <<'EOF' > logs/llm_depths_summary.txt
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
for d in ("p2c", "c2p"):
    print(f"== {d} (all relations) ==")
    for path in sorted(out.glob(f"metrics_{d}*.json"), key=lambda p: json.load(open(p))["topn"]):
        m = json.load(open(path))
        if path.name != f"metrics_{d}.json" and not path.name.startswith(f"metrics_{d}_top"):
            continue
        depth = m["topn"]
        for name in m["llm_rerank"]:
            for label in (name, f"{name}+LLM"):
                r = m["metrics"]["all"][label]
                cols = "  ".join(f"{k}={v:.4f}" for k, v in r.items())
                print(f"depth {depth:>3}  queries {m['n_queries']['all']:>4}  {label:14s} {cols}")
EOF
say "all done (local=$local_done remote=$remote_done; 1 = ok, 2 = gave up); summary: logs/llm_depths_summary.txt"
