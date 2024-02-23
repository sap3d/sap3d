import torch

# 定义你的文件路径
file_path = '/home/xinyang/sap3d/camerabooth/experiments_GSO_add_view_3/Dino_3/2024-02-22T23-04-00_config_Dino_3_view_3-1-1000-1e-06-0.1-1e-07/evaluation/epoch_99/results.tar'

# 使用torch.load读取文件
data = torch.load(file_path)

# 打印读取的数据，确保加载成功
print(data)

# 例如，访问并打印字典中的一些元素
print('Elevation Prediction:', data['elevation_pred'])
print('Elevation Ground Truth:', data['elevation_gt'])
# 以此类推，根据需要处理其他键值