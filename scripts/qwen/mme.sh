#!/bin/bash

python -m llava.eval.qwen_model_vqa_loader \
    --model-path Qwen-VL-Chat \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/Qwen-VL-Chat/memvr.jsonl \
    --temperature 0 \
    --cuda-device 'cuda:0' \
    --apply-memvr 'memvr' \
    --retracing-ratio 0.28 \
    --entropy-threshold 0.75 \
    --max-new-tokens 1 \
    --starting-layer 9 \
    --ending-layer 16 \

cd ./playground/data/eval/MME
python convert_answer_to_mme.py --experiment Qwen-VL-Chat/memvr \

cd eval_tool
python calculation.py --results_dir answers/Qwen-VL-Chat/memvr