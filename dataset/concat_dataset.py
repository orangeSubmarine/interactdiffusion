from .catalog import DatasetCatalog
from ldm.util import instantiate_from_config
import torch
"""
拼接多个数据集的类构造函数
"""
class ConCatDataset():
    def __init__(self, dataset_name_list, ROOT, train=True, repeats=None):
        self.datasets = []
        cul_previous_dataset_length = 0
        offset_map = []  # 索引偏移量映射
        which_dataset = []  # 样本来源数据集标记

        #重复次数处理
        if repeats is None: 
            repeats = [1] * len(dataset_name_list)
        else:
            assert len(repeats) == len(dataset_name_list)

        Catalog = DatasetCatalog(ROOT)
        for dataset_idx, (dataset_name, yaml_params) in enumerate(dataset_name_list.items()):
            repeat = repeats[dataset_idx]

            dataset_dict = getattr(Catalog, dataset_name)

            target = dataset_dict['target']
            params = dataset_dict['train_params'] if train else dataset_dict['val_params']
            if yaml_params is not None:
                params.update(yaml_params)
            dataset = instantiate_from_config(dict(target=target, params=params))

            self.datasets.append(dataset)
            for _ in range(repeat):
                offset_map.append(torch.ones(len(dataset)) * cul_previous_dataset_length)
                which_dataset.append(torch.ones(len(dataset)) * dataset_idx)
                cul_previous_dataset_length += len(dataset)
        offset_map = torch.cat(offset_map, dim=0).long()
        self.total_length = cul_previous_dataset_length

        self.mapping = torch.arange(self.total_length) - offset_map
        self.which_dataset = torch.cat(which_dataset, dim=0).long()

    def total_images(self):
        count = 0
        for dataset in self.datasets:
            print(dataset.total_images())
            count += dataset.total_images()
        return count

    def __getitem__(self, idx):
        dataset = self.datasets[self.which_dataset[idx]]
        return dataset[self.mapping[idx]]

    def __len__(self):
        return self.total_length
