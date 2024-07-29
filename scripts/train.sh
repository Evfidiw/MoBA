cd "$(dirname "$0")"/..

python3 main.py \
--model CLIP \
--fusion att \
--weight_decay 0.05 \
--train_batch_size 64 \
--dev_batch_size 64 \
--learning_rate 5e-4 \
--clip_learning_rate 1e-6 \
--layers 3 \
--max_grad_norm 5 \
--dropout_rate 0.1 \
--optimizer_name adam \
--text_size 512 \
--image_size 768 \
--warmup_proportion 0.2 \
--device 0 \
--device 0 \
--seed 3407 \
--num_experts 8 \
--layers 3 \
--num_validation_steps 2000 \
--early_stop 5 \
--num_train_epochs 5 \
--text_name text_json_clean \
--train \
--test \
