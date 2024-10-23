import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.llava_arch import LlavaMetaForCausalLM

from PIL import Image
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# MemVR
from memvr import apply_memvr_qwen

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    device= args.cuda_device
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    checkpoint = os.path.join(os.path.dirname(__file__), '..', '..', args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map=device, trust_remote_code=True).eval()


    model.generation_config = GenerationConfig.from_pretrained(checkpoint, trust_remote_code=True)
    model.generation_config.top_p = 0.01


    model.transformer.h[0].mlp.starting_layer = args.starting_layer
    model.transformer.h[0].mlp.ending_layer = args.ending_layer
    model.transformer.h[0].mlp.vision_retracing_method = args.vision_retracing
    model.transformer.h[0].mlp.entropy_threshold = args.entropy_threshold
    for layer in range(32):
        model.transformer.h[layer].mlp.retracing_ratio = args.retracing_ratio

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # MemVR
    if args.apply_memvr == 'memvr':
        apply_memvr_qwen(
            self=model,
            starting_layer=args.starting_layer,
            ending_layer=args.ending_layer,
            entropy_threshold=args.entropy_threshold,
            retracing_ratio=args.retracing_ratio
        )
    else:
        model.transformer.h[0].mlp.apply_memvr = False

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        images = args.image_folder + "/" + image_file



        with torch.inference_mode():
            query = f'<img>{images}</img>\n{cur_prompt}'
            outputs, _ = model.chat(tokenizer, query=query, history=None, temperature=args.temperature, do_sample=False)


        print ("\n Outputs: ", outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    
    # MemVR
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--vision-retracing", type=str, default="default")
    parser.add_argument("--retracing-ratio", type=float, default=0.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.75)
    parser.add_argument("--starting-layer", type=int, default=5)
    parser.add_argument("--ending-layer", type=int, default=16)
    parser.add_argument("--apply-memvr", type=str, default='default')
    args = parser.parse_args()

    eval_model(args)
