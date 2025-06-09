import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import logging
import random
from utils import MetricsTop, dict_to_str
logger = logging.getLogger('MMSA')


class MAMLTrainer:
    def __init__(self, args):
        self.args = args
        self.inner_lr = 0.01
        self.outer_lr = 0.001
        self.num_tasks = 3
        self.num_inner_steps = 1
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        
        # 定义三类任务的缺失率配置
        self.task_configs = [
            {
                'mr_index': 1,  # 对应 mr=0.1
                'miss_2': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][0],  # 两个模态缺失的比例
                'miss_1': [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][0],  # 一个模态缺失的比例
                'task_id': 0,
                'support_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0},
                'query_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0}
            },
            {
                'mr_index': 4,  # 对应 mr=0.4
                'miss_2': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][3],
                'miss_1': [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][3],
                'task_id': 1,
                'support_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0},
                'query_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0}
            },
            {
                'mr_index': 7,  # 对应 mr=0.7
                'miss_2': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][6],
                'miss_1': [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][6],
                'task_id': 2,
                'support_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0},
                'query_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0}
            }
        ]
        
    def get_modality_num(self, task, set_type, total_set_size, current_batch_size):
        """
        根据整个数据集的进度决定使用几个模态
        Args:
            task: 当前任务配置
            set_type: 'support_stats' 或 'query_stats'
            total_set_size: 当前集合的总样本数
            current_batch_size: 当前批次大小
        """
        stats = task[set_type]
        stats['total'] += current_batch_size

        # 计算目标的两模态和单模态缺失数量
        target_miss_two = int(total_set_size * task['miss_2'])
        target_miss_one = int(total_set_size * task['miss_1'])
        # 基于整体进度决定使用几个模态
        if stats['miss_two'] < target_miss_two:
            stats['miss_two'] += current_batch_size
            return 1
        elif stats['miss_one'] < target_miss_one:
            stats['miss_one'] += current_batch_size
            return 2
        else:
            return 3


    def update_task_configs(self):
        """在每个epoch开始时更新任务配置的缺失率"""
        # 为每个任务类别随机选择mr_index
        low_mr_index = random.choice([1, 2, 3])
        medium_mr_index = random.choice([4, 5])
        high_mr_index = random.choice([6, 7])
        
        # 更新任务配置
        for task in self.task_configs:
            if task['task_id'] == 0:  # 低缺失率任务
                task['mr_index'] = low_mr_index
                task['miss_2'] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][low_mr_index - 1]
                task['miss_1'] = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][low_mr_index - 1]
            elif task['task_id'] == 1:  # 中缺失率任务
                task['mr_index'] = medium_mr_index
                task['miss_2'] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][medium_mr_index - 1]
                task['miss_1'] = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][medium_mr_index - 1]
            else:  # 高缺失率任务
                task['mr_index'] = high_mr_index
                task['miss_2'] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][high_mr_index - 1]
                task['miss_1'] = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][high_mr_index - 1]
            
            # 重置统计信息
            task['support_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}
            task['query_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}

    def reset_missing_counts(self):
        """在每个epoch开始时重置计数器并更新任务配置"""
        # 首先更新任务配置
        #self.update_task_configs()
        
        # 然后重置计数器（原有功能保持不变）
        for task in self.task_configs:
            task['support_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}
            task['query_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}
        
    def calculate_set_sizes(self, dataloader):
        """计算support和query集的总大小"""
        total_samples = sum(batch['vision'].size(0) for batch in dataloader)
        support_size = int(total_samples * 0.25)  # 25% 用于support
        query_size = total_samples - support_size
        return support_size, query_size
            
    def inner_loop_step(self, model, support_data, num_modal):
        """内循环步骤 - 简化版本，直接使用传入的num_modal"""
        params = self.get_inner_loop_params(model)
        
        for _ in range(self.num_inner_steps):
            outputs = self.forward_data(model, support_data, num_modal=num_modal, params=params)
            loss = self.compute_loss(outputs, support_data['labels']['M'].to(self.args.device))
            params = self.update_parameters(model, loss, params, self.inner_lr, False)
        
        return params

    def outer_loop_step(self, model, query_data, num_modal, params):
        """外循环步骤 - 简化版本，直接使用传入的num_modal"""
        outputs = self.forward_data(model, query_data, num_modal=num_modal, params=params)
        return self.compute_loss(outputs, query_data['labels']['M'].to(self.args.device))
    
    def split_batch(self, batch_data):
        """将batch分为support和query集，确保每个集合至少有一个样本"""
        split_ratio = 0.25  # 25%用于support set
        support_data = {}
        query_data = {}
        
        min_batch_size = 4  # 确保至少有一个样本用于support，三个用于query
        if any(isinstance(v, torch.Tensor) and v.size(0) < min_batch_size for v in batch_data.values()):
            raise ValueError(f"Batch size must be at least {min_batch_size}")
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                split_idx = max(1, int(value.size(0) * split_ratio))  # 确保至少有一个样本
                support_data[key] = value[:split_idx].clone()
                query_data[key] = value[split_idx:].clone()
            elif isinstance(value, dict):
                support_data[key] = {}
                query_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        split_idx = max(1, int(sub_value.size(0) * split_ratio))
                        support_data[key][sub_key] = sub_value[:split_idx].clone()
                        query_data[key][sub_key] = sub_value[split_idx:].clone()
            else:
                support_data[key] = value
                query_data[key] = value
        
        return support_data, query_data



    def sample_tasks(self, mode='train'):
        """
        采样任务，每次返回三种不同缺失率的任务
        
        Args:
            mode (str): 'train'/'val'/'test'，用于区分训练和测试阶段
            
        Returns:
            list: 包含三种任务配置的列表
        """
        if mode == 'train':
            # 训练时返回所有三种任务
            return self.task_configs
        else:
            # 测试时也返回三种任务以全面评估性能
            return self.task_configs
    
    def get_inner_loop_params(self, model):
        """获取内循环可训练的参数"""
        params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:  # 只获取需要训练的参数
                params[name] = param.clone().detach().requires_grad_(True)
        return params

    def forward_data(self, model, data, num_modal=3, params=None):
        """使用指定参数进行前向传播"""
        vision = data['vision'].to(self.args.device)
        audio = data['audio'].to(self.args.device)
        text = data['text'].to(self.args.device)
        if params is None:
            # 确保所有参数可训练
            for param in model.parameters():
                param.requires_grad = True
            return model(text, audio, vision, num_modal=num_modal)
        
        # 保存原始参数
        orig_params = {}
        for name, param in model.named_parameters():
            orig_params[name] = param.data.clone()
            if name in params:
                param.data.copy_(params[name].data)
            param.requires_grad = True  # 确保参数可训练
        
        
        try:
            outputs = model(text, audio, vision, num_modal=num_modal)
        finally:
            # 恢复原始参数
            for name, param in model.named_parameters():
                param.data.copy_(orig_params[name])
        return outputs

    
    def update_parameters(self, model, loss, params=None, step_size=None, first_order=False):
        """更新参数"""
        if step_size is None:
            step_size = self.inner_lr
            
        if params is None:
            # 使用模型当前参数，并确保它们可训练
            params = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:  # 只包含需要梯度的参数
                    param.requires_grad_(True)  # 确保参数可训练
                    params[name] = param
        
        # 确保所有参数都需要梯度
        trainable_params = []
        param_names = []
        for name, param in params.items():
            if isinstance(param, torch.Tensor) and param.requires_grad:
                trainable_params.append(param)
                param_names.append(name)
        
        if not trainable_params:  # 如果没有可训练的参数，返回原始参数
            return params
            
        # 计算梯度
        grads = torch.autograd.grad(
            loss,
            trainable_params,
            create_graph=not first_order,
            allow_unused=True
        )
        
        # 更新参数
        updated_params = OrderedDict()
        for name, param in params.items():
            if name in param_names:
                idx = param_names.index(name)
                grad = grads[idx]
                if grad is not None:
                    updated_params[name] = param - step_size * grad
                else:
                    updated_params[name] = param.clone().detach().requires_grad_(True)
            else:
                updated_params[name] = param.clone().detach().requires_grad_(True)
        
        return updated_params



    def compute_loss(self, outputs, labels):
        """计算总损失"""
        task_loss = self.criterion(outputs['M'], labels)
        aux_loss = 0.1 * sum(outputs[k] for k in ['loss_score_l', 'loss_score_v', 'loss_score_a', 'loss_rec'])
        return task_loss + aux_loss

    def train_epoch(self, model, task_dataloaders):
        """
        训练一个epoch
        Args:
            model: 要训练的模型
            task_dataloaders: 包含三个任务的DataLoader列表
        Returns:
            float: 训练损失
        """
        model.train()
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.outer_lr)
        total_loss = 0
        n_batches = 0
        
        # 获取最短的dataloader长度，确保任务平衡
        min_batches = min(len(loader) for loader in task_dataloaders)
        # 计算每个任务的数据集总大小
        task_set_sizes = {}
        for task_idx, loader in enumerate(task_dataloaders):
            support_size, query_size = self.calculate_set_sizes(loader)
            task_set_sizes[task_idx] = {
                'support': support_size,
                'query': query_size
            }
        # 重置所有任务的缺失计数
        self.reset_missing_counts()
        
        # 创建每个任务的迭代器
        task_iterators = [iter(loader) for loader in task_dataloaders]
        
        with tqdm(total=min_batches, desc="Training Epoch", leave=False) as pbar:
            for batch_idx in range(min_batches):
                # 获取每个任务的batch数据
                task_batches = []
                for task_iter in task_iterators:
                    try:
                        batch = next(task_iter)
                        task_batches.append(batch)
                    except StopIteration:
                        task_iter = iter(task_dataloaders[task_iterators.index(task_iter)])
                        batch = next(task_iter)
                        task_batches.append(batch)
                
                outer_loss = torch.tensor(0., device=self.args.device)
                
                # 对每个任务进行训练
                for task_idx, (task, batch_data) in enumerate(zip(self.task_configs, task_batches)):
                    try:
                        # 分割support和query集
                        support_data, query_data = self.split_batch(batch_data)
                        # 根据整个数据集进度决定模态数量
                        support_modal = self.get_modality_num(
                            task, 'support_stats',
                            task_set_sizes[task_idx]['support'],
                            support_data['vision'].size(0)
                        )
                        
                        # 内循环适应
                        fast_weights = self.inner_loop_step(model, support_data, support_modal)
                        
                        # 外循环评估
                        query_modal = self.get_modality_num(
                            task, 'query_stats',
                            task_set_sizes[task_idx]['query'],
                            query_data['vision'].size(0)
                        )
                        
                        task_loss = self.outer_loop_step(model, query_data, support_modal, fast_weights)
                        outer_loss += task_loss
                        
                    except ValueError as e:
                        logger.warning(f"Error in task {task['task_id']} batch {batch_idx}: {str(e)}")
                        continue
                
                # 计算平均损失并更新
                outer_loss.div_(len(self.task_configs))
                meta_optimizer.zero_grad()
                outer_loss.backward()      
                        
                if self.args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(
                        [param for param in model.parameters() if param.requires_grad],
                        self.args.grad_clip
                    )
                
                meta_optimizer.step()
                
                total_loss += outer_loss.item()
                n_batches += 1
                
                pbar.set_postfix({'loss': f"{outer_loss.item():.4f}"})
                pbar.update(1)
                
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
            
        return total_loss / n_batches if n_batches > 0 else float('inf')

    def evaluate(self, model, task_dataloaders, mode="VAL"):
        """
        测试函数，确保每个任务都进行正确的任务适应
        Args:
            model: 模型
            task_dataloaders: 包含三个任务的DataLoader列表
            mode: "VAL" 或 "TEST"
        """    
        print(f"Testing on {mode} set...")
        model.train()  # MAML中测试时也使用train模式
        total_loss = 0
        
        # 为每个任务维护单独的预测结果
        task_predictions = {
            'low_mr': {'y_pred': [], 'y_true': []},     
            'medium_mr': {'y_pred': [], 'y_true': []},  
            'high_mr': {'y_pred': [], 'y_true': []}     
        }
        
        # 获取最短的dataloader长度
        min_batches = min(len(loader) for loader in task_dataloaders)
        # 计算每个任务的数据集大小
        task_set_sizes = {}
        for task_idx, loader in enumerate(task_dataloaders):
            support_size, query_size = self.calculate_set_sizes(loader)
            task_set_sizes[task_idx] = {
                'support': support_size,
                'query': query_size
            }
        # 重置所有任务的缺失计数
        self.reset_missing_counts()
        
        # 创建每个任务的迭代器
        task_iterators = [iter(loader) for loader in task_dataloaders]
        
        with tqdm(total=min_batches, desc=f'{mode} Evaluation', leave=False) as pbar:
            for batch_idx in range(min_batches):
                batch_loss = torch.tensor(0., device=self.args.device)
                
                # 获取每个任务的batch数据
                task_batches = []
                for task_iter in task_iterators:
                    try:
                        batch = next(task_iter)
                        task_batches.append(batch)
                    except StopIteration:
                        task_iter = iter(task_dataloaders[task_iterators.index(task_iter)])
                        batch = next(task_iter)
                        task_batches.append(batch)
                
                # 对每个任务分别进行测试
                for task_idx, (task, batch_data) in enumerate(zip(self.task_configs, task_batches)):
                    try:
                        # 分割当前任务的数据为support和query
                        support_data, query_data = self.split_batch(batch_data)
                        # 在support set上进行任务适应
                        adapted_params = None
                        for _ in range(self.num_inner_steps):
                            support_modal = self.get_modality_num(
                                task, 'support_stats',
                                task_set_sizes[task_idx]['support'],
                                support_data['vision'].size(0)
                            )
                            
                            support_outputs = self.forward_data(
                                model, 
                                support_data,
                                num_modal=support_modal,
                                params=adapted_params
                            )
                            
                            support_loss = self.compute_loss(
                                support_outputs,
                                support_data['labels']['M'].to(self.args.device)
                            )
                            
                            # 更新适应参数
                            adapted_params = self.update_parameters(
                                model=model,
                                loss=support_loss,
                                params=adapted_params,
                                step_size=self.inner_lr,
                                first_order=True
                            )
                        
                        # 在query set上评估
                        with torch.no_grad():
                            query_modal = self.get_modality_num(
                                task, 'query_stats',
                                task_set_sizes[task_idx]['query'],
                                query_data['vision'].size(0)
                            )
                            
                            query_outputs = self.forward_data(
                                model,
                                query_data,
                                num_modal=support_modal,
                                params=adapted_params
                            )
                            
                            query_labels = query_data['labels']['M'].to(self.args.device)
                            task_loss = self.compute_loss(query_outputs, query_labels)
                            batch_loss += task_loss
                            
                            # 存储每个任务的预测结果
                            task_type = ['low_mr', 'medium_mr', 'high_mr'][task_idx]
                            task_predictions[task_type]['y_pred'].append(query_outputs['M'].cpu())
                            task_predictions[task_type]['y_true'].append(query_labels.cpu())
                    
                    except ValueError as e:
                        logger.warning(f"Error in task {task['task_id']} batch {batch_idx}: {str(e)}")
                        continue
                
                # 更新总损失和进度条
                total_loss += batch_loss.item()
                pbar.set_postfix({'loss': f"{batch_loss.item():.4f}"})
                pbar.update(1)
                
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
        # 计算并返回结果
        results = {}
        F1_scores = []
        for task_type in task_predictions:
            if task_predictions[task_type]['y_pred']:
                pred = torch.cat(task_predictions[task_type]['y_pred'])
                true = torch.cat(task_predictions[task_type]['y_true'])
                task_metrics = self.metrics(pred, true)
                for metric_name, value in task_metrics.items():
                    results[f"{task_type}_{metric_name}"] = value
                    
                if f"F1_score" in task_metrics:
                    F1_scores.append(task_metrics["F1_score"])
        if F1_scores:
            results["F1_score"] = sum(F1_scores) / len(F1_scores)
        # 添加总体损失
        results["Loss"] = total_loss / min_batches
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(results)}")
        return results