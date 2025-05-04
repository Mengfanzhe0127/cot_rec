#!/bin/bash

# 检查是否提供了PID和指令
if [ $# -lt 2 ]; then
  echo "Usage: $0 <pid> <command1> [command2 ...]"
  exit 1
fi

PID=$1
shift # 移除PID，剩下的是要执行的指令

# 检查PID进程是否存在
while kill -0 $PID 2>/dev/null; do
  echo "进程 $PID 还在运行，等待中..."
  sleep 1 # 每秒检查一次
done

echo "进程 $PID 已结束，开始执行指定指令..."

python memoCRS_main.py --method only-label
python memoCRS_main.py --method memory-label


echo "所有指令已执行完毕"
