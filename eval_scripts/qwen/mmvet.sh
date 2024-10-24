#!/bin/bash

model='Qwen-VL-Chat'
method='memvr'
python -m llava.eval.qwen_model_vqa \
    --model-path $model \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.28 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 9 \
    --ending-layer 16 \

mkdir -p ./playground/data/eval/mm-vet/results/$model
python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$model/${method}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$model/${method}.json \