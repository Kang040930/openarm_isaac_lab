#!/bin/bash
# Auto-run training and save to sequentially numbered folder
# Usage: bash trainings/run.sh [description]

RUNS=$(ls -d /home/kk/openarm_isaac_lab-1/trainings/run_* 2>/dev/null | wc -l)
if ls -d /home/kk/openarm_isaac_lab-1/trainings/run_* 2>/dev/null 1>&2; then
    true
else
    RUNS=0
fi
RUN_NUM=$((RUNS + 1))
RUN_NAME="run_$(printf '%03d' $RUN_NUM)${1:+_$1}"
SAVE_DIR="/home/kk/openarm_isaac_lab-1/trainings/$RUN_NAME"

echo "=== Training: $RUN_NAME ==="
echo "Saving to: $SAVE_DIR"

rm -rf /home/kk/openarm_isaac_lab-1/logs/rsl_rl/openarm_bi_lift/
find /home/kk/openarm_isaac_lab-1/source/openarm -name "__pycache__" -exec rm -rf {} + 2>/dev/null

python ./scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-OpenArm-Bi-v0 \
    --headless \
    --num_envs 4096 \
    --video \
    --video_interval 4800 \
    --video_length 300

LATEST_LOG=$(ls -dt /home/kk/openarm_isaac_lab-1/logs/rsl_rl/openarm_bi_lift/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    mkdir -p "$SAVE_DIR"/{models,videos,logs}
    cp "$LATEST_LOG"model_*.pt "$SAVE_DIR/models/" 2>/dev/null
    cp -r "$LATEST_LOG"videos/* "$SAVE_DIR/videos/" 2>/dev/null
    cp "$LATEST_LOG"events.out.* "$SAVE_DIR/logs/" 2>/dev/null
    echo "=== Saved to $SAVE_DIR ==="
fi
