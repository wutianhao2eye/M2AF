# **Meta-Learning for Incomplete Multimodal Sentiment Analysis**

## Usage

### Prerequisites
- Python 3.8
- PyTorch 1.9.0
- CUDA 11.4

### Datasets
Reference: [IMDer](https://github.com/mdswyz/IMDer/tree/main) .

You only need to download the aligned data. You can put the downloaded datasets into `dataset/` directory. 

### Pretrained models

Before running missing cases, you should obtain the weights pretrained by [IMDer](https://github.com/mdswyz/IMDer/tree/main) at different missing rates.
You can put the pretrained models  weights into `IMDER_trained_model/` directory.

(i.e., dataset=mosi 、MR=0.4、seed=1115 pretrained models for IMDER_trained_model\mosi\pt_mosi_mr4_seed1115\).

You can also directly download the  pretrained model on IMDer under different missing rate [download](https://drive.google.com/drive/folders/1rYqT-lR-TF73pEkfT_csA-_YeHpEK84d?usp=drive_link) .

You need to place the models trained by [IMDer](https://github.com/mdswyz/IMDer/tree/main) under different missing rates into `IMDER_trained_model/ ` directory.

## 🌈 How to train and evaluate on datasets

Training:

Running the following command:
```
python train_maml.py --dataset-name mosi --seed 1114 --mr 0.5  #for mosi  missing rate 0.5

python train_maml.py --dataset-name mosei --seed 1114 --mr 0.1  #for mosei missing rate 0.1
```

Evaluation:

Run evaluation scripts: [`run_test_mosi.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_audio.sh) or [`run_test_mosei.sh`](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs/blob/master/Uni_MoE/eval_speech.sh) using ```bash run_test_mosi.sh``` ```bash run_test_mosei.sh``` or run the following commands to eval:

```
#for mosi  missing rate 0.5
python test_mosi.py --model_save_path ./models/pt_mosi_mr5_seed1114_maml/M2AF-mosi.pth  --seed 1114 --output_file ./test_results/mosi_mr5_seed1114_maml_30%.txt --support_size 24 --split_ratio 0.5    

#for mosei  missing rate 0.1
python test_mosei.py --model_save_path ./models/pt_mosei_mr1_seed1114_maml/M2AF-mosei.pth  --seed 1114 --output_file ./test_results/mosei_mr1_seed1114_maml_3%.txt --support_size 40 --split_ratio 0.10  
```

### Citation

If you find the code helpful in your research or work, please cite the following paper.
```

```
