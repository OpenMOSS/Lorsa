#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: bash test.sh <start> <end>"
    exit 1
fi

start=$1
end=$2
gpu_count=4
TOTAL_TASKS=$((end - start))

# 将当前任务编号写入文件，作为共享计数器
echo 0 > /tmp/current_task.txt

# 定义任务执行函数
run_task() {
    local gpu_id=$1
    local task_id=$2
    local actual_id=$((start + task_id))
    echo "[GPU $gpu_id] Running task $actual_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python /inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/zfhe/jx_projects/Lorsa/monosemanticity/k/code/cal_pattern_mse_dfa_lorsa_head.py --layer 5 --head_index $actual_id
}

# 定义每个 GPU 的 worker，循环领取任务直到结束
worker() {
    local gpu_id=$1
    while true; do
        # 使用文件锁保护任务计数器的读写
        exec 200>/tmp/task_lock
        flock 200

        current_task=$(cat /tmp/current_task.txt)
        if [ $current_task -ge $TOTAL_TASKS ]; then
            # 释放锁并退出循环
            flock -u 200
            break
        fi

        task_id=$current_task
        next_task=$((current_task + 1))
        echo $next_task > /tmp/current_task.txt

        # 释放锁后再执行任务
        flock -u 200

        run_task $gpu_id $task_id
    done
}

# 启动四个 worker，每个分配一个 GPU
for gpu_id in $(seq 0 $((gpu_count - 1))); do
    worker $gpu_id &
done

wait