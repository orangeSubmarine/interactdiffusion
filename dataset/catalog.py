import os 
"""数据集目录配置类的初始化代码,集中管理不同数据集的路径配置"""
class DatasetCatalog:
    def __init__(self, ROOT):
        self.HicoDetHOI = {
            "target": "dataset.hico_dataset.HICODataset",
            "train_params":dict(
                dataset_path=os.path.join(ROOT,'hico_det_clip'), #将根目录与数据集子目录拼接
            ),
        }

        self.VisualGenome = {
            "target": "dataset.hico_dataset.HICODataset",
            "train_params": dict(
                dataset_path=os.path.join(ROOT, 'vg_clip'),
            ),
        }


