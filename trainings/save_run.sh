#!/bin/bash
# Usage: bash trainings/save_run.sh <run_name> <log_dir>
# Example: bash trainings/save_run.sh joint_pos_v1 logs/rsl_rl/openarm_bi_lift/2026-05-28_15-xx-xx

RUN_NAME=$1
LOG_DIR=$2
SAVE_DIR="/home/kk/openarm_isaac_lab-1/trainings/$RUN_NAME"

mkdir -p "$SAVE_DIR"/{models,videos,logs}
cp "$LOG_DIR"/model_*.pt "$SAVE_DIR/models/" 2>/dev/null
cp -r "$LOG_DIR"/videos/* "$SAVE_DIR/videos/" 2>/dev/null
cp "$LOG_DIR"/events.out.* "$SAVE_DIR/logs/" 2>/dev/null

echo "Saved to $SAVE_DIR"
ls "$SAVE_DIR"
