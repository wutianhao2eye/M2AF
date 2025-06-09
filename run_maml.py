import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from data_loader import MMDataLoader, create_maml_dataloaders
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import imder

logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # Same as before
    log_file_path = Path(log_dir) / f"M2AF-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def _run(args, num_workers=4, is_tune=False, from_sena=False, trainer_class=None):
    # 加载数据
    dataloader = MMDataLoader(args, num_workers)
    
    # 初始化模型
    model = getattr(imder, 'IMDER')(args)
    model = model.cuda()

    if trainer_class is not None:
        trainer = trainer_class(args)
    else:
        trainer = ATIO().getTrain(args)

    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(args.model_save_path))
    
    # 测试
    test_dataloaders = create_maml_dataloaders(
                dataloader['test'],
                20,
                4,
                sample_ratio=0.1
            )
    results = trainer.evaluate(model, test_dataloaders, mode="TEST")
    #results = trainer.do_test(model, dataloader['test'], mode="TEST")

    # 清理内存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return results

def IMDER_run(
        model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
        tune_times=500, feature_T="", feature_A="", feature_V="",
        model_save_dir="./models", res_save_dir="./results", log_dir="./logs",
        gpu_ids=[0], num_workers=4, verbose_level=1, mode='train',
        trainer_class=None,  # 新增trainer_class参数
        pretrained_model_path=None,  # 新增参数
    ):
    # 初始化
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    # 配置文件处理
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    
    # 创建必要的目录
    model_save_dir = Path(model_save_dir)
    res_save_dir = Path(res_save_dir)
    log_dir = Path(log_dir)
    for dir_path in [model_save_dir, res_save_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 设置logger
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)

    # 获取配置
    args = get_config_regression(model_name, dataset_name, config_file)
    args.mode = mode
    args['model_save_path'] = model_save_dir / f"M2AF-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    
    # 新增参数
    args['model_path'] = pretrained_model_path
    
    if config:
        args.update(config)

    # 准备结果保存
    res_save_dir = res_save_dir / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(args)
    # 多次运行，记录结果
    model_results = []
    for i, seed in enumerate(seeds):
        print(f'Running with seed {seed} ({i + 1}/{len(seeds)})')
        logger.info(f'Running with seed {seed} ({i + 1}/{len(seeds)})')
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune, trainer_class=trainer_class)
        model_results.append(result)

    # 保存结果
    criterions = list(model_results[0].keys())
    csv_file = res_save_dir / f"{dataset_name}.csv"
    
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)

    # 计算并保存统计结果
    res = [model_name]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")

    return model_results