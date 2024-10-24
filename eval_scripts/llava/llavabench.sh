#!/bin/bash

model="llava-v1.5-7b"
method="memvr"
python -m llava.eval.llava_model_vqa \
    --model-path $model \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$model/${method}.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1024 \
    --starting-layer 5 \
    --ending-layer 16 \