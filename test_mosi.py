import torch
from collections import OrderedDict
from tqdm import tqdm
import logging
import torch
from pathlib import Path
from data_loader import MMDataLoader  # Ensure this module exists
from utils import setup_seed  # Ensure this module exists
from trains.singleTask.model import imder  # Ensure this module exists
from argparse import Namespace
import numpy as np
from datetime import datetime
import logging
from collections import OrderedDict
from utils import MetricsTop, dict_to_str  # Ensure this module exists
from tqdm import tqdm
import random
from argparse import ArgumentParser, Namespace
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MMSA')

def parse_args():
    parser = ArgumentParser(description='Test MOSI model with specified parameters')
    parser.add_argument('--model_save_path', type=str, required=True,
                      help='Path to the saved model file')
    parser.add_argument('--seed', type=int, default=1114,
                      help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='./test_results/mosi_mr4_seed1114_maml.txt',
                      help='Output file name for saving results')
    parser.add_argument('--support_size', type=int, default=40,
                      help='support_size')
    parser.add_argument('--split_ratio', type=float, default=0.1,  # 添加一个小数参数
                        help='split_ratio')
    return parser.parse_args()

def initialize_args(mr_value, model_save_path):
    """Initialize testing parameters, dynamically set missing rate (MR)."""
    return {
        'model_name': 'imder',
        'dataset_name': 'mosi',
        'featurePath': 'dataset/MOSI/aligned_50.pkl',
        'feature_dims': [768, 5, 20],
        'train_samples': 1284,
        'num_classes': 3,
        'language': 'en',
        'KeyEval': 'F1_score',

        # 通用
        'need_data_aligned': True,
        'need_model_aligned': True,
        'early_stop': 20,
        'use_bert': True,
        'use_finetune': True,
        'attn_mask': True,
        'update_epochs': 1,

        # 来自 imder.datasetParams.mosi
        'attn_dropout_a': 0.2,
        'attn_dropout_v': 0.0,
        'relu_dropout': 0.0,
        'embed_dropout': 0.0,
        'res_dropout': 0.0,
        'dst_feature_dim_nheads': [32, 8],
        'batch_size': 32,
        'learning_rate': 0.002,
        'nlevels': 4,
        'conv1d_kernel_size_l': 3,
        'conv1d_kernel_size_a': 3,
        'conv1d_kernel_size_v': 3,
        'text_dropout': 0.0,
        'attn_dropout': 0.0,
        'output_dropout': 0.0,
        'grad_clip': 1.0,
        'patience': 24,
        'weight_decay': 0.005,
        'transformers': 'bert',
        'pretrained': 'bert-base-uncased',

        # 测试模式 & 缺失率
        'mode': 'test',
        'mr': mr_value,

        # 你需要将模型的实际保存路径改成你训练 MOSI 的 pth
        'model_save_path': Path(model_save_path),

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'train_mode': 'regression',

        # 为空即可
        'feature_T': '',
        'feature_A': '',
        'feature_V': '',
    }


class SingleTaskMAMLTester:
    def __init__(self, args):
        self.args = args
        self.inner_lr = 0.001
        self.num_inner_steps = 1
        self.criterion = torch.nn.L1Loss() if args.train_mode == 'regression' else torch.nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        self.task_config = {
            'mr_index': int(args.mr * 10 - 1),
            'miss_2': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0][int(args.mr * 10 - 1)],
            'miss_1': [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0][int(args.mr * 10 - 1)],
            'task_id': 0,
            'support_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0},
            'query_stats': {'miss_two': 0, 'miss_one': 0, 'total': 0}
        }

    def get_inner_loop_params(self, model):
        """获取内循环可训练的参数"""
        params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                params[name] = param.clone().detach().requires_grad_(True)
        return params

    def update_parameters(self, model, loss, params=None, step_size=None, first_order=False):
        if step_size is None:
            step_size = self.inner_lr
            
        if params is None:
            params = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.requires_grad_(True)
                    params[name] = param
        
        for param in params.values():
            if param.grad is not None:
                param.grad.zero_()
        
        loss.backward(retain_graph=True)  # 使用retain_graph=True
        
        updated_params = OrderedDict()
        for name, param in params.items():
            if param.grad is not None:
                updated_params[name] = param - step_size * param.grad
            else:
                updated_params[name] = param.clone().detach().requires_grad_(True)
        
        return updated_params

    def forward_data(self, model, data, num_modal=3, params=None):
        """使用指定参数进行前向传播"""
        vision = data['vision'].to(self.args.device)
        audio = data['audio'].to(self.args.device)
        text = data['text'].to(self.args.device)
        if params is None:
            for param in model.parameters():
                param.requires_grad = True
            return model(text, audio, vision, num_modal=num_modal)
        
        orig_params = {}
        for name, param in model.named_parameters():
            orig_params[name] = param.data.clone()
            if name in params:
                param.data.copy_(params[name].data)
            param.requires_grad = True
        
        try:
            outputs = model(text, audio, vision, num_modal=num_modal)
        finally:
            for name, param in model.named_parameters():
                param.data.copy_(orig_params[name])
                
        return outputs

    def compute_loss(self, outputs, labels):
        """计算总损失"""
        task_loss = self.criterion(outputs['M'], labels)
        aux_loss = 0.1 * sum(outputs[k] for k in ['loss_score_l', 'loss_score_v', 'loss_score_a', 'loss_rec'])
        return task_loss + aux_loss


    def reset_missing_counts(self):
        """Reset counters at the start of each epoch."""
        self.task_config['support_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}
        self.task_config['query_stats'] = {'miss_two': 0, 'miss_one': 0, 'total': 0}

    def calculate_set_sizes(self, dataloader):
        """Calculate total sizes for support and query sets."""
        total_samples = sum(batch['vision'].size(0) for batch in dataloader)
        support_size = int(total_samples * 0.25)  # 25% for support
        query_size = total_samples - support_size
        return support_size, query_size

    def split_batch(self, batch_data, split_ratio=0.25):
        """Split batch into support and query sets."""
        support_data = {}
        query_data = {}
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                split_idx = max(1, int(value.size(0) * split_ratio))
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

    def sample_support_set(self, train_dataloader, split_ratio, support_size=40):
        """
        Randomly sample multiple batches from the training set as the support set,
        applying the missing rate settings.
        support_size: Number of batches in the support set
        """
        num_batches = len(train_dataloader)
        if support_size <= 0:
            raise ValueError(f"Support size must be a positive integer. Got: {support_size}")
        if support_size > num_batches:
            support_size = num_batches
            print(f"Support size is larger than the number of batches in the training set. ")
        support_batches = []
        # Randomly sample support_size batches
        sampled_batches = random.sample(list(train_dataloader), support_size)
        self.reset_missing_counts()
        
        # Calculate the number of each missing mode
        num_modal1 = int(np.round(support_size * self.task_config['miss_2']))
        num_modal2 = int(np.round(support_size * self.task_config['miss_1']))
        num_modal3 = support_size - num_modal1 - num_modal2
        
        # Create modal assignments list
        modal_assignments = [1] * num_modal1 + [2] * num_modal2 + [3] * num_modal3
        for idx, batch_data in enumerate(sampled_batches):
            support_batch, _ = self.split_batch(batch_data, split_ratio)  # All as support
            num_modal = modal_assignments[idx]
            support_batches.append((support_batch, num_modal))
        
        return support_batches

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        miss_one, miss_two = 0, 0

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    miss_2 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
                    miss_1 = [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0]
                    if miss_two / (np.round(len(dataloader) / 10) * 10) < miss_2[int(self.args.mr * 10 - 1)]:  # missing two modal
                        outputs = model(text, audio, vision, num_modal=1)
                        miss_two += 1
                    elif miss_one / (np.round(len(dataloader) / 10) * 10) < miss_1[int(self.args.mr * 10 - 1)]:  # missing one modal
                        outputs = model(text, audio, vision, num_modal=2)
                        miss_one += 1
                    else:  # no missing
                        outputs = model(text, audio, vision, num_modal=3)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())

                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results

    def do_test_with_fixed_support_and_fixed_mr(self, model, split_ratio, train_dataloader, test_dataloader, mode="TEST", support_size=5):
        model.train()
        support_batches = self.sample_support_set(train_dataloader, split_ratio, support_size=support_size)
        original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=self.inner_lr)

        for support_batch, num_modal in support_batches:
            support_outputs = self.forward_data(model, support_batch, num_modal=num_modal)
            support_loss = self.compute_loss(support_outputs, 
                                        support_batch['labels']['M'].to(self.args.device))
            inner_optimizer.zero_grad()
            support_loss.backward()
            inner_optimizer.step()
            
        adapted_state_dict = model.state_dict()
        model.eval()
        results = self.do_test(model, test_dataloader, mode=mode)
        model.load_state_dict(original_state_dict)
        
        return results

def test_model(seed_value, mr_value, model_save_path, support_size, split_ratio):
    """Test the model using a specific seed and MR value."""
    setup_seed(seed_value)
    args = Namespace(**initialize_args(mr_value, model_save_path))
    dataloaders = MMDataLoader(vars(args), num_workers=4)
    model = imder.IMDER(args)
    model.load_state_dict(torch.load(args.model_save_path, map_location=args.device))
    model = model.to(args.device)
    tester = SingleTaskMAMLTester(args)
    results = tester.do_test_with_fixed_support_and_fixed_mr(model, split_ratio, dataloaders['train'], dataloaders['test'], mode="TEST", support_size=support_size)
    return results

def calculate_average(results_list):
    """Calculate average metrics over multiple runs."""
    keys = results_list[0].keys()
    avg_results = {}
    
    for key in keys:
        avg_results[key] = np.mean([result[key] for result in results_list])
    
    return avg_results

def save_results_to_file(results_dict, filename='single_task_maml_results.txt'):
    """Save test results to a file."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'a') as f:
        for mr_value, (results, avg_results) in results_dict.items():
            f.write(f"Timestamp: {current_time}\n")
            f.write(f"MR = {mr_value}\n")
            f.write("Individual Results:\n")
            for i, result in enumerate(results):
                f.write(f"  Seed {1111 + i}: {result}\n")
            f.write(f"Average Results: {avg_results}\n\n")

if __name__ == "__main__":
    args = parse_args()
    init_args = Namespace(**initialize_args(0.1, args.model_save_path))
    all_results = {}
    
    # Test different MR values from 0.1 to 0.7
    for mr_value in np.arange(0.1, 0.8, 0.1):
        mr_value = round(mr_value, 1)
        print(f"开始测试 MR = {mr_value}...")
        results_list = []
        results = test_model(args.seed, mr_value, args.model_save_path, args.support_size, args.split_ratio)  # support_size can be adjusted as needed
        results_list.append(results)
        avg_results = calculate_average(results_list)
        all_results[mr_value] = (results_list, avg_results)
        save_results_to_file({mr_value: (results_list, avg_results)}, args.output_file)
        torch.cuda.empty_cache()
    
    print(f"所有 MOSI 测试结果已保存到 '{args.output_file}'。")
