cd "$(dirname "$0")"/..

python3 main.py \
--device 0 \
--model_path "openai/clip-vit-large-patch14" \
--fusion 'att' \
--train_batch_size 64 \
--dev_batch_size 64 \
--text_size 768 \
--image_size 768 \
--seed 3407 \
--num_experts 4 \
--layers 3 \
--num_validation_steps 2000 \
--early_stop 5 \
--num_train_epochs 10 \
--text_name text_json_clean \
--train \
--test \
