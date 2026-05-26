#!/bin/bash

set -e

ORIGINAL_ARGS=("$@")

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DATASET_BASE_URL="${NANOCHAT_DATASET_BASE_URL:-https://huggingface.co/datasets/jrast/clean-text-v1/resolve/main}"
export NANOCHAT_DATASET_MAX_SHARD="${NANOCHAT_DATASET_MAX_SHARD:-auto}"
export NANOCHAT_DATASET_DIR_NAME="${NANOCHAT_DATASET_DIR_NAME:-full_5m_may_2026}"
NPROC_PER_NODE=8
CORE_METRIC_EVERY=200
SAVE_EVERY=200
USE_TMUX=1
AUTO_SHUTDOWN=1
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-speedrun_full_5m_may_2026}"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --gpus=*)
            NPROC_PER_NODE="${1#*=}"
            shift
            ;;
        --single-gpu)
            NPROC_PER_NODE=1
            shift
            ;;
        --core-every)
            CORE_METRIC_EVERY="$2"
            shift 2
            ;;
        --core-every=*)
            CORE_METRIC_EVERY="${1#*=}"
            shift
            ;;
        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --save-every=*)
            SAVE_EVERY="${1#*=}"
            shift
            ;;
        --no-tmux)
            USE_TMUX=0
            shift
            ;;
        --no-shutdown)
            AUTO_SHUTDOWN=0
            shift
            ;;
        --tmux-session)
            TMUX_SESSION_NAME="$2"
            shift 2
            ;;
        --tmux-session=*)
            TMUX_SESSION_NAME="${1#*=}"
            shift
            ;;
        *)
            echo "Usage: $0 [--gpus 1|8] [--single-gpu] [--core-every N] [--save-every N] [--no-tmux] [--no-shutdown] [--tmux-session NAME]" >&2
            exit 1
            ;;
    esac
done

if [ "$NPROC_PER_NODE" != "1" ] && [ "$NPROC_PER_NODE" != "8" ]; then
    echo "NPROC_PER_NODE must be 1 or 8" >&2
    exit 1
fi

if ! [[ "$CORE_METRIC_EVERY" =~ ^-?[0-9]+$ ]]; then
    echo "CORE_METRIC_EVERY must be an integer" >&2
    exit 1
fi

if ! [[ "$SAVE_EVERY" =~ ^-?[0-9]+$ ]]; then
    echo "SAVE_EVERY must be an integer" >&2
    exit 1
fi

if [ -z "$TMUX" ] && [ "${NANOCHAT_IN_TMUX:-0}" != "1" ] && [ "$USE_TMUX" = "1" ]; then
    command -v tmux &> /dev/null || (apt-get update && apt-get install -y tmux)
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "tmux session already exists: $TMUX_SESSION_NAME" >&2
        echo "Attach with: tmux attach -t $TMUX_SESSION_NAME" >&2
        exit 1
    fi
    TMUX_CMD="cd $(printf "%q" "$PWD") && NANOCHAT_IN_TMUX=1 bash $(printf "%q" "$0")"
    for arg in "${ORIGINAL_ARGS[@]}"; do
        TMUX_CMD="$TMUX_CMD $(printf "%q" "$arg")"
    done
    tmux new-session -d -s "$TMUX_SESSION_NAME" "$TMUX_CMD"
    echo "Started tmux session: $TMUX_SESSION_NAME"
    echo "Attach with: tmux attach -t $TMUX_SESSION_NAME"
    exit 0
fi

shutdown_node() {
    if [ "$AUTO_SHUTDOWN" != "1" ]; then
        return
    fi
    sync
    if command -v runpodctl &> /dev/null; then
        runpodctl stop pod
    elif command -v shutdown &> /dev/null; then
        shutdown -h now
    elif command -v poweroff &> /dev/null; then
        poweroff
    else
        echo "No shutdown command found" >&2
    fi
}

apt-get update
apt-get install -y python3-dev build-essential rsync vim tmux

mkdir -p /workspace/.cache

if [ -d /root/.cache ] && [ ! -L /root/.cache ]; then
    rsync -a /root/.cache/ /workspace/.cache/
    rm -rf /root/.cache
fi

ln -sfn /workspace/.cache /root/.cache

mkdir -p $NANOCHAT_BASE_DIR

export PATH="$HOME/.local/bin:$PATH"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=12 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN --core-metric-every=$CORE_METRIC_EVERY --save-every=$SAVE_EVERY --wandb-save-checkpoints

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN

shutdown_node
