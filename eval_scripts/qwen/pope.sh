#!/bin/bash

model="Qwen-VL-Chat"
method="memvr"
data="coco_pope_test" # change it to gqa_pope_test or aokvqa_pope_test if needed
python -m llava.eval.qwen_model_vqa_loader \
    --model-path $model \
    --question-file ./playground/data/eval/pope/${data}.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$model/$data/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.28 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/${data}.jsonl \
    --result-file ./playground/data/eval/pope/answers/$model/$data/${method}.jsonl \