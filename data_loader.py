import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']
logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.raw_text = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        if self.args['feature_T'] != "":
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if 'use_bert' in self.args and self.args['use_bert']:
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2]
        if self.args['feature_A'] != "":
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2]
        if self.args['feature_V'] != "":
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2]

        self.labels = {
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args['need_data_aligned']:
            if self.args['feature_A'] != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args['feature_V'] != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncate(self):
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }
    # print("in",args['batch_size'])
    # args.batch_size = max(12, args.batch_size - args.batch_size % 3)  # 确保是3的倍数且至少为12
    # print(f'after batch',args['batch_size'])
    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader










class MAMLDataset(Dataset):
    def __init__(self, original_dataset, task_id, sample_ratio=0.01, seed=42):
        """
        Args:
            original_dataset: 原始数据集
            task_id: 任务ID (0, 1, 2)
            sample_ratio: 采样比例
            seed: 随机种子
        """
        self.original_dataset = original_dataset
        self.task_id = task_id
        
        # 设置随机种子确保可重复性
        np.random.seed(seed + task_id)
        
        # 计算要采样的数据量
        total_samples = len(original_dataset)
        print(f"Task {task_id}: {total_samples} samples")
        num_samples = max(int(total_samples * sample_ratio), 1)  # 确保至少有一个样本
        
        # 随机采样索引
        self.indices = np.random.choice(
            total_samples, 
            size=num_samples, 
            replace=False
        )

    def __getitem__(self, idx):
        # 使用采样的索引获取原始数据
        original_idx = self.indices[idx]
        sample = self.original_dataset[original_idx]
        
        # 确保返回一个字典
        if not isinstance(sample, dict):
            sample = {'data': sample}
        
        # 添加任务ID信息
        sample['task_id'] = self.task_id
        return sample

    def __len__(self):
        return len(self.indices)

def create_maml_dataloaders(original_dataloader, batch_size, num_workers, sample_ratio=0.01):
    """
    为MAML创建三个任务的数据加载器
    
    Args:
        original_dataloader: 原始数据加载器
        batch_size: 批次大小
        num_workers: 工作进程数
        sample_ratio: 采样比例，默认0.01
        
    Returns:
        list: 包含三个任务数据加载器的列表
    """
    task_dataloaders = []
    
    for task_id in range(3):
        # 为每个任务创建采样数据集
        task_dataset = MAMLDataset(
            original_dataloader.dataset,
            task_id=task_id,
            sample_ratio=sample_ratio
        )
        
        # 创建数据加载器
        task_dataloader = torch.utils.data.DataLoader(
            task_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        
        task_dataloaders.append(task_dataloader)
    
    return task_dataloaders