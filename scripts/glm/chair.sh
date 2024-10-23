#!/bin/bash

model="glm-4v-9b"
method="memvr"
python -m llava.eval.glm_model_vqa_loader_chair \
    --model-path $model \
    --question-file ./playground/data/eval/chair/annotations/instances_val2014.json \
    --image-folder ./playground/data/eval/chair/val2014 \
    --answers-file ./playground/data/eval/chair/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.26 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \