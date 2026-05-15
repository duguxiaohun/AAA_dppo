#!/bin/bash
export PYTHONPATH="/home/codon/github/AAA_dppo:$PYTHONPATH"
SCRIPT_PATH="script/offline_carla_collector.py"
PYTHON_PROCESS="python3 $SCRIPT_PATH"

MAX_RUNS=30
run_count=0

while [ $run_count -lt $MAX_RUNS ]; do
    # 启动脚本
    echo "[$(date)] 启动脚本... (运行次数: $((run_count+1))/$MAX_RUNS)"
    python3 "$SCRIPT_PATH" &
    PID=$!

    # 等待进程结束
    while ps -p $PID > /dev/null; do
        sleep 1
    done

    echo "[$(date)] 脚本执行结束"
    sleep 1  # 防止CPU占用过高

    run_count=$((run_count+1))
done

echo "[$(date)] 已完成 $MAX_RUNS 次运行，脚本退出"