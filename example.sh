

TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python train_scaff.py \
    dataset mutag semi_ratio 0.9 lm.train.batch_size 64 lm.train.epochs 20 lm.model.name roberta-base;


