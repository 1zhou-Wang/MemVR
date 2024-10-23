#!/bin/bash

python -m llava.eval.llava_model_vqa_loader \
    --model-path llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b/memvr.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.12 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 5 \
    --ending-layer 16 \

cd ./playground/data/eval/MME
python convert_answer_to_mme.py --experiment llava-v1.5-7b/memvr \

cd eval_tool
python calculation.py --results_dir answers/llava-v1.5-7b/memvr