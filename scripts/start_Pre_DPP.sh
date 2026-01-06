#!/bin/bash
set -e

ROOT="/data/FL/Semol"
SCRIPT="$ROOT/Pre_DPP.py"
LOGDIR="$ROOT/scripts/logs"
PID_FILE="$ROOT/scripts/Pre_DPP.pid"

mkdir -p "$LOGDIR"

echo "=== 启动 DDP 训练 (torchrun) ==="
echo "时间: $(date)"
echo "工作目录: $ROOT"
echo "入口脚本: $SCRIPT"

if [ ! -f "$SCRIPT" ]; then
  echo "❌ 找不到入口脚本: $SCRIPT"
  echo "请确认 Pre_DPP.py 是否在 $ROOT 下"
  exit 1
fi

RUNNING_PID=$(pgrep -f "torchrun.*${SCRIPT}" || true)
if [ -n "$RUNNING_PID" ]; then
  echo "⚠️ 已在运行！PID: $RUNNING_PID"
  ps -p "$RUNNING_PID" -o pid,etime,cmd
  exit 0
fi

# 线程限制
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export OUTDATED_IGNORE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=WARN

LOG_FILE="$LOGDIR/ddp_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo "开始 torchrun ..."

cd "$ROOT"

setsid nohup torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
  > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown || true

echo "✅ 已后台启动，PID=$PID"
echo "PID 文件: $PID_FILE"
echo "查看日志: tail -f $LOG_FILE"
echo "停止训练: kill \$(cat $PID_FILE)"
