# Optimal Temperature
Follows https://arxiv.org/pdf/2605.08505.

## Usage
```
git clone karpathy's nanogpt
prepare the shakespeare data following karpathy's instructions
replace model.py with the one in this repo
then run
python3.10 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

## Results

baseline:
```
step 2000: train loss 1.7648, val loss 1.8857
```

optimal per token temperature across 3 seeds:
```
step 2000: train loss 1.6809, val loss 1.8445
step 2000: train loss 1.6796, val loss 1.8676
step 2000: train loss 1.7051, val loss 1.8550
```
