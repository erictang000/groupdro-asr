#!/bin/sh
python main.py \
    --use_group_dro True \
    --train_set 1htrain \
    --steps 1000 \
    --num_workers 4 \
    --model_id openai/whisper-small \
    --train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --eval_batch_size 32 \
    --languages bas kin lug nso nya ssw swa tsn ven xho zul

