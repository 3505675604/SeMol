#!/bin/bash
set -e

ROOT="/data/FL/MolCL-SP"
SCRIPT="$ROOT/Pre_dan_duo.py"

LOGDIR="$ROOT/scripts/logs"
PID_FILE="$ROOT/scripts/Pre_dan_duo.pid"

mkdir -p "$LOGDIR"

echo "=== 启动 Pre_dan_duo.py ==="
echo "时间: $(date)"
echo "工作目录: $ROOT"
echo "程序文件: $SCRIPT"

# 检查入口脚本是否存在
if [ ! -f "$SCRIPT" ]; then
  echo "❌ 找不到程序文件: $SCRIPT"
  exit 1
fi

# 检查是否已在运行（用绝对路径更准）
RUNNING_PID=$(pgrep -f "python.*${SCRIPT}" || true)
if [ -n "$RUNNING_PID" ]; then
  echo "⚠️  程序已在运行！PID: $RUNNING_PID"
  echo "运行时间: $(ps -p $RUNNING_PID -o etime= 2>/dev/null || echo '未知')"
  exit 0
fi

# 生成日志文件名（放到 scripts/logs 下）
LOG_FILE="$LOGDIR/Pre_dan_duo_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"

# ===== 线程限制（推荐：训练/推理场景设为 1）=====
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# （可选）关掉 outdated 版本检查刷屏
export OUTDATED_IGNORE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "🚀 启动程序中..."

# 切到项目根目录，避免相对路径/导入问题
cd "$ROOT"

# setsid + nohup：关闭 VS/断开 SSH 不会断
setsid nohup python "$SCRIPT" > "$LOG_FILE" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PID_FILE"
disown || true

echo "✅ 启动成功！PID: $PID"

sleep 3
if ps -p $PID > /dev/null 2>&1; then
  echo "📊 进程状态: 运行中"
  echo "查看日志: tail -f $LOG_FILE"
  echo "停止程序: kill \$(cat $PID_FILE)"
else
  echo "⚠️  进程可能已退出，请检查日志:"
  tail -50 "$LOG_FILE"
  exit 1
fi
