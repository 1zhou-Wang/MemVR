#!/bin/bash

model="llava-v1.5-7b"
method="memvr"
python -m llava.eval.llava_model_vqa_loader_chair \
    --model-path $model \
    --question-file ./playground/data/eval/chair/annotations/instances_val2014.json \
    --image-folder ./playground/data/eval/chair/val2014 \
    --answers-file ./playground/data/eval/chair/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.31 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \
