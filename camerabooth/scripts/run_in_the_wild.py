import subprocess
import time
import os
import multiprocessing

available_gpus         = [7]  # 设置总共的GPU数量
max_parallel_processes = len(available_gpus)

# 检查GPU内存是否满足需求
def check_gpu_memory(gpu_id):
    result = subprocess.check_output(
        f"nvidia-smi --query-gpu=memory.free --format=csv,noheader --id={gpu_id}", shell=True
    ).decode('utf-8')
    memory = int(result.strip().split()[0])
    return memory >= 35000

# 获取所有空闲的GPU
def get_free_gpus():
    free_gpus = []
    for gpu_id in available_gpus:
        if check_gpu_memory(gpu_id):
            free_gpus.append(gpu_id)
    return free_gpus

# 子进程函数
def run_script(script, gpu_id, semaphore):
    print(f"Running {script} on GPU {gpu_id}")
    subprocess.call(["bash", script, str(gpu_id)])
    semaphore.release()

if __name__ == "__main__":
    # 配置
    # todo 👇 debug param name，add new name (refer to edit_config_param.py)
    script_paths = [
        "scripts/launchs/CAPTURE_experiments_in_the_wild",
    ]
    # todo 👆 debug param name，add new name (refer to edit_config_param.py)

    # 获取所有空闲的GPU
    free_gpus = get_free_gpus()
    scripts = []

    # 遍历脚本路径，将脚本添加到列表中
    for script_path in script_paths:
        scripts.extend(
            [
                f'{script_path}/{i}' for i in sorted(os.listdir(script_path))
            ]
        )

    # 创建信号量，限制并行执行的子进程数量
    semaphore = multiprocessing.Semaphore(max_parallel_processes)

    processes = []

    # 分配GPU并启动子进程
    for script in scripts:
        while not free_gpus:
            print("No free GPUs available. Waiting for a GPU to become free...")
            time.sleep(100)
            free_gpus = get_free_gpus()

        gpu_id = free_gpus.pop(0)  # 分配一个可用的GPU
        semaphore.acquire()  # 获取信号量，控制并行进程数量
        # 启动子进程，并将其添加到进程列表中
        process = multiprocessing.Process(target=run_script, args=(script, gpu_id, semaphore))
        process.start()
        processes.append(process)

    # 等待所有子进程完成
    for process in processes:
        process.join()
        print(f"Script {process} on GPU {gpu_id} has finished.")

