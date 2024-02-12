import os
import random
import shutil

if __name__ == '__main__':
    src_folders = [
        '../dataset/data/all_demo/GSO_demo',
        '../dataset/data/all_demo/ABO_demo',
    ]
    
    train_view_set = [
        
    ]
    
    for src_folder in src_folders:
        src_name = src_folder.split('/')[-1]
        output_folder = f'../dataset/data/{src_name}'
        all_class_paths = [f'{src_folder}/{i}' for i in os.listdir(src_folder)]
        for class_path in all_class_paths:
            class_name = class_path.split('/')[-1]
            output_path = f'{output_folder}/{class_name}'
            # 选index
            elevation_60 = [0,1,2,3,4,5,6]
            elevation_90 = [7,8,9,10,11,12,13]
            
            # 随机选，但azimuth不能重复
            if random.random() < 0.5:
                select_ind0 = random.sample(elevation_60, 1)[0]
                elevation_90.pop(select_ind0)
                select_ind1, select_ind2 = random.sample(elevation_90, 2)
            else:
                select_ind0, select_ind1 = random.sample(elevation_60, 2)
                del_ele0, del_ele1 = elevation_90[elevation_60.index(select_ind0)], elevation_90[elevation_60.index(select_ind1)]
                elevation_90.remove(del_ele0)
                elevation_90.remove(del_ele1)
                select_ind2 = random.sample(elevation_90, 1)[0]
            
            # elevation_120 = [14,15,16,17,18,19,20]
            # elevation_120.pop(elevation_60.index(select_ind0))
            # elevation_120.pop(elevation_90.index(select_ind1))
            # select_ind2 = random.sample(elevation_120, 1)[0]

            train_views = [
                select_ind0, select_ind1, select_ind2
            ]
            os.makedirs(f'{output_path}/images', exist_ok=True)
            os.makedirs(f'{output_path}/poses', exist_ok=True)
            for i in range(3):
                # 复制图像
                shutil.copyfile(f'{class_path}/images/{train_views[i]:03d}.png', f'{output_path}/images/{i:03d}.png')
                # 复制pose
                shutil.copyfile(f'{class_path}/poses/{train_views[i]:03d}.npy', f'{output_path}/poses/{i:03d}.npy')
            # 复制query class
            # shutil.copytree(f'{class_path}/query_class', f'{output_path}/query_class')
            # shutil.copy(f'{class_path}/valid_paths.json', output_path)