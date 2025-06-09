from trains.singleTask import IMDER
from run_maml import IMDER_run
from maml_trainer import MAMLTrainer
import torch
import logging
from tqdm import tqdm
import logging
from utils import MetricsTop, dict_to_str
import os
from data_loader import create_maml_dataloaders
logger = logging.getLogger('MMSA')

class IMDER_MAML():
    def __init__(self, args):
        self.args = args
        self.maml_trainer = MAMLTrainer(args)
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.base_trainer = IMDER(args)

    def do_train(self, model, dataloader, return_epoch_results=False):
        # 创建三个任务的数据加载器
        task_dataloaders = create_maml_dataloaders(
            dataloader['train'],
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sample_ratio=0.2 #0.1 0.2 0.3 for mosi
            #sample_ratio=0.02 #0.01 0.02 0.03 for mosei
         )
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        # 加载预训练模型
        try:
            pretrained_path = os.path.join(self.args.model_path, f'imder-{self.args.dataset_name}.pth')
            logger.info(f'Loading pretrained model from {pretrained_path}')
            pretrained_dict = torch.load(pretrained_path)
            model.load_state_dict(pretrained_dict)
            logger.info('Successfully loaded pretrained model')
        except Exception as e:
            logger.error(f'Failed to load pretrained model: {str(e)}')
            logger.error('A pretrained model is required but was not found or failed to load. Exiting.')
            raise e
        
        while True:
            epochs += 1
            self.maml_trainer.update_task_configs()
            # MAML训练
            train_loss = self.maml_trainer.train_epoch(model, task_dataloaders)

            # 验证和测试
            val_dataloaders = create_maml_dataloaders(
                dataloader['valid'],
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sample_ratio=1
            )
            
            val_results = self.evaluate(model, val_dataloaders, mode="VAL")
            
            cur_valid = val_results[self.args.KeyEval]
            
            # 记录训练进度
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(val_results)}"
            )
            

            model_save_dir = os.path.dirname(self.args.model_save_path)
            os.makedirs(model_save_dir, exist_ok=True)
            model_save_path = os.path.join(model_save_dir, str(epochs) + '.pth')
            torch.save(model.state_dict(), model_save_path)
            # 更新最佳模型
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            
            # 记录每个epoch的结果
            if return_epoch_results:
                epoch_results['train'].append({"Loss": train_loss})
                epoch_results['valid'].append(val_results)
                #epoch_results['test'].append(test_results)
            
            # 早停检查
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
    
    def evaluate(self, model, dataloader, mode="VAL"):
        return self.maml_trainer.evaluate(model, dataloader, mode=mode)
# 修改运行入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='IMDER MAML Training')
    parser.add_argument('--dataset-name', type=str, default='mosi', help='dataset name, mosi or mosei')
    parser.add_argument('--seed', type=int, default=1115, help='random seed')
    parser.add_argument('--mr', type=float, default=0.4, help='missing ratio (e.g., 0.4)')
    cmd_args = parser.parse_args()

    dataset_name = cmd_args.dataset_name
    seed = cmd_args.seed
    mr_float = cmd_args.mr
    mr_int = int(mr_float * 10)

    model_path_part = f'pt_{dataset_name}_mr{mr_int}_seed{seed}'

    IMDER_run(
        model_name='imder',
        dataset_name=dataset_name,
        seeds=[seed],
        trainer_class=IMDER_MAML,  # 使用MAML训练器
        pretrained_model_path=f'./IMDER_trained_model/{dataset_name}/{model_path_part}',  # 指定预训练模型路径
        model_save_dir=f'./models/{model_path_part}_maml',
        res_save_dir=f'./results/result_{dataset_name}_mr{mr_int}_seed{seed}_maml',
        log_dir=f'./logs/log_{dataset_name}_mr{mr_int}_seed{seed}_maml'
    )