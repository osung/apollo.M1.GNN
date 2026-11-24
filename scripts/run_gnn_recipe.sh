#!/usr/bin/env bash
# GNN training recipe that avoids the "warm vs cold company" shortcut of uniform random
# negatives: similarity edges for message passing + mined hard negatives + both directions
# (the settings of scripts/run_sweep.py). Usage: scripts/run_gnn_recipe.sh [layer-type ...]
#   EPOCHS=100 scripts/run_gnn_recipe.sh lightgcn      # train to 100 epochs
#   EPOCHS=150 RESUME=1 scripts/run_gnn_recipe.sh hgt  # continue from the last saved model
# A checkpoint (z, model, held-out metrics) is written every CKPT_EVERY epochs, so the
# loss/metric curve shows whether training has converged and a run can be extended.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# SUBGRAPH=<dir>: train on the reduced graph of scripts/build_subgraph.py (its paths.yaml /
# graph.yaml point every cache and output into that directory).
P=${SUBGRAPH:-data/processed}
cfg_flag=(); [ -n "${SUBGRAPH:-}" ] && cfg_flag=(--paths "$P/paths.yaml" --graph-cfg "$P/graph.yaml")
paths_flag=(); [ -n "${SUBGRAPH:-}" ] && paths_flag=(--paths "$P/paths.yaml")
export PYTHONUNBUFFERED=1
THREADS=${THREADS:-16}
EPOCHS=${EPOCHS:-100}
CKPT_EVERY=${CKPT_EVERY:-10}
CKPT=$P/checkpoints
HARD_RATIO=${HARD_RATIO:-0.8}        # share of the negatives taken from the mined hard pool
WARM_NEG_RATIO=${WARM_NEG_RATIO:-0}  # share of the random negatives drawn from warm nodes
MASK_BATCH_EDGES=${MASK_BATCH_EDGES:-0}
mask_flag=(); [ "$MASK_BATCH_EDGES" = 1 ] && mask_flag=(--mask-batch-edges)
WITH_AUX=${WITH_AUX:-}               # e.g. "keyword inherited": derived message-passing edges
MP_WEIGHTS=${MP_WEIGHTS:-0}          # 1 = scale messages by the per-relation edge weight
AMP=${AMP:-none}                     # bf16 halves activation memory (HGT with many relations)
DEVICE=${DEVICE:-cuda}
FANOUTS=${FANOUTS:-}                 # e.g. "10 5": mini-batch training with neighbor sampling
BATCH_SIZE=${BATCH_SIZE:-}
DROP_RELATIONS=${DROP_RELATIONS:-}   # e.g. "patent": ablation without that observed relation
TAG_SUFFIX=${TAG_SUFFIX:-}
abl_flag=(); [ -n "$DROP_RELATIONS" ] && abl_flag=(--drop-relations $DROP_RELATIONS)
[ -n "$TAG_SUFFIX" ] && abl_flag+=(--tag-suffix "$TAG_SUFFIX")
mb_flag=(); [ -n "$FANOUTS" ] && mb_flag=(--fanouts $FANOUTS)
[ -n "$BATCH_SIZE" ] && mb_flag+=(--batch-size "$BATCH_SIZE")
[ -n "${NEIGHBOR_SELECTION:-}" ] && mb_flag+=(--neighbor-selection "$NEIGHBOR_SELECTION")
aux_flag=(); [ -n "$WITH_AUX" ] && aux_flag=(--with-aux $WITH_AUX)
# WITH_SIM=0: no similarity edges in message passing (ablation; hard negatives are unchanged)
sim_flag=(--with-similarity --sim-path $P/sim_edges_top10.npz); [ "${WITH_SIM:-1}" = 0 ] && sim_flag=()
mpw_flag=(); [ "$MP_WEIGHTS" = 1 ] && mpw_flag=(--mp-edge-weights)
for kind in $WITH_AUX; do
  [ -f $P/aux_edges_${kind}.npz ] || $PY scripts/build_aux_edges.py --kind "$kind" --device "$DEVICE" "${cfg_flag[@]}"
done
[ -f $P/sim_edges_top100.npz ] || $PY scripts/build_similarity.py "${cfg_flag[@]}" --topk 100 --threads "$THREADS" --output $P/sim_edges_top100.npz
[ -f $P/sim_edges_top10.npz ]  || $PY scripts/build_similarity.py "${cfg_flag[@]}" --topk 10  --threads "$THREADS" --output $P/sim_edges_top10.npz
for d in p2c c2p; do
  [ -f $P/hard_negatives_${d}_clean.npz ] || $PY scripts/build_hard_negatives.py "${paths_flag[@]}" --direction $d \
      --sim-path $P/sim_edges_top100.npz --exclude-sim-path $P/sim_edges_top10.npz \
      --output $P/hard_negatives_${d}_clean.npz
done
for layer in "${@:-lightgcn hgt}"; do
  for l in $layer; do
    resume=()
    if [ "${RESUME:-0}" = 1 ]; then
      last=$(ls -t $CKPT/model_${l}_*${RESUME_TAG:-}*.pt 2>/dev/null | head -1)
      [ -n "$last" ] && resume=(--resume "$last")
    fi
    $PY scripts/train_gnn.py "${cfg_flag[@]}" --layer-type "$l" --device cuda --num-neg 20 --no-normalize \
        --hard-ratio "$HARD_RATIO" --warm-neg-ratio "$WARM_NEG_RATIO" "${mask_flag[@]}" \
        "${aux_flag[@]}" "${mpw_flag[@]}" "${mb_flag[@]}" "${abl_flag[@]}" --amp-dtype "$AMP" \
        --hard-neg-path $P/hard_negatives_p2c_clean.npz \
        --hard-neg-path-c2p $P/hard_negatives_c2p_clean.npz --p2c-weight 1.0 --c2p-weight 1.0 \
        "${sim_flag[@]}" --epochs "$EPOCHS" \
        --checkpoint-every "$CKPT_EVERY" --checkpoint-dir $CKPT --save-model-ckpt \
        --eval-at-checkpoint --direction both "${resume[@]}" 2>&1 | tee -a "logs/train_${l}_recipe.log"
  done
done
