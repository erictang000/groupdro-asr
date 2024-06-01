#!/bin/sh
python main_group_sources.py \
    --use_group_dro False \
    --train_set 10mintrain \
    --steps 0 \
    --num_workers 4 \
    --model_id openai/whisper-base \
    --train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --eval_batch_size 32 \
    --processor_language english \
    --sources commonvoice fleurs nchlt voxforge mls voxpopuli swc M-AILABS LAD