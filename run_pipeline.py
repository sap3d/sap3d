import numpy as np
import subprocess
import time
import torch
import glob
import os
import multiprocessing

# ! uasge demo
# python run_pipeline.py --object_type GSO_demo_tab3_row3
# python run_pipeline.py --object_type GSO_demo

# available_gpus         = [0,1,2,5,7]  
available_gpus         = [0,1,2,3,4,5,6,7]  
max_parallel_processes = len(available_gpus)

def check_gpu_memory(gpu_id):
    result = subprocess.check_output(
        f"nvidia-smi --query-gpu=memory.free --format=csv,noheader --id={gpu_id}", shell=True
    ).decode('utf-8')
    memory = int(result.strip().split()[0])
    return memory >= 36000

def get_free_gpus():
    free_gpus = []
    for gpu_id in available_gpus:
        if check_gpu_memory(gpu_id):
            free_gpus.append(gpu_id)
    return free_gpus

def run_script(object_type, object_name, view_num, gpu_id, semaphore):
    print(f"Running {object_type} {object_name} on GPU {gpu_id}")
    subprocess.call(["sh", "run_pipeline.sh", object_type, object_name, str(view_num), str(gpu_id)])
    semaphore.release()

def get_configures():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_type', type=str, default='XINYANG')
    return parser.parse_args()

def find_last_log(dir_path):
    timestamp_dirs = glob.glob(os.path.join(dir_path, "*"))
    timestamp_dirs.sort(key=os.path.getmtime, reverse=True)
    if timestamp_dirs:
        latest_dir = timestamp_dirs[0]
        return latest_dir
    return None

if __name__ == "__main__":
    opt = get_configures()

    train_view_setting = [3]

    args_list = []
    for train_view in train_view_setting:
        free_gpus = get_free_gpus()

        class_names = os.listdir(f'dataset/data/train/{opt.object_type}')

        for class_name in sorted(class_names):
            all_view = 0
            for i in range(train_view):
                if os.path.exists(f'dataset/data/train/{opt.object_type}/{class_name}/images/{i:03d}.png'):
                    all_view += 1

            if all_view == train_view:
                args_list.append(
                    [
                        opt.object_type, class_name, train_view
                    ]
                )
                    
    print(len(args_list))

    semaphore = multiprocessing.Semaphore(max_parallel_processes)
    processes = []

    print('Avaiable Case:', len(args_list))
    for args in args_list:
        while not free_gpus:
            print("No free GPUs available. Waiting for a GPU to become free...")
            time.sleep(1000)
            free_gpus = get_free_gpus()

        gpu_id = free_gpus.pop(0)  
        semaphore.acquire()  
        object_type, object_name, num_view = args
        process = multiprocessing.Process(target=run_script, args=(object_type, object_name, num_view, gpu_id, semaphore))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        print(f"Script {process} on GPU {gpu_id} has finished.")
