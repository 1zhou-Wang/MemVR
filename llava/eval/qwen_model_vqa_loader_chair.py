import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from PIL import Image
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib as mpl
import json


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, process_anyres_image
from torch.utils.data import Dataset, DataLoader
from llava.model.llava_arch import LlavaMetaForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import math

# MemVR
from memvr import apply_memvr_glm

parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model", default="llava-v1.5-7b")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.", default=0)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="annotations", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")

parser.add_argument("--beam", type=int)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="llava-v1.5-7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-folder", type=str, default="./playground/data/eval/pope/val2014")
parser.add_argument("--question-file", type=str, default="./playground/data/eval/pope/llava_pope_test.jsonl")
parser.add_argument("--answers-file", type=str, default="./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl")
parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--chunk-idx", type=int, default=0)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max-new-tokens", type=int, default=512)

# MemVR
parser.add_argument("--cuda-device", type=str, default="cuda:0")
parser.add_argument("--vision-retracing", type=str, default="default")
parser.add_argument("--retracing-ratio", type=float, default=0.0)
parser.add_argument("--entropy-threshold", type=float, default=0.75)
parser.add_argument("--starting-layer", type=int, default=5)
parser.add_argument("--ending-layer", type=int, default=16)
parser.add_argument("--apply-memvr", type=str, default='default')
args = parser.parse_args()


 



# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
# args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
# cfg = Config(args)
# setup_seeds(cfg)
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

device= args.cuda_device
model_path = 'Qwen-VL-Chat'
model_name = get_model_name_from_path(model_path)

checkpoint = os.path.join(os.path.dirname(__file__), '..', '..', model_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map=device, trust_remote_code=True).eval()

if args.apply_memvr == 'memvr':
    apply_memvr_glm(
        self=model,
        starting_layer=args.starting_layer,
        ending_layer=args.ending_layer,
        entropy_threshold=args.entropy_threshold,
        retracing_ratio=args.retracing_ratio
    )

input_file = 'playground/data/eval/chair/shuffled_img_files.txt' # Please provide the path to the images
with open(input_file, 'r') as f:
    img_files = f.read().splitlines()

with open(args.question_file, 'r') as f:
    lines = f.readlines()
coco_anns = json.loads(lines[0])

img_dict = {}

categories = coco_anns["categories"]
category_names = [c["name"] for c in categories]
category_dict = {int(c["id"]): c["name"] for c in categories}

for img_info in coco_anns["images"]:
    img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

for ann_info in coco_anns["annotations"]:
    img_dict[ann_info["image_id"]]["anns"].append(
        category_dict[ann_info["category_id"]]
    )


os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

for img_id in range(len(img_files)):
    if img_id == 500:
        break
    img_file = img_files[img_id]
    img_id = int(img_file.split(".jpg")[0][-6:])
    img_info = img_dict[img_id]
    assert img_info["name"] == img_file
    img_anns = set(img_info["anns"])
    img_save = {}
    img_save["image_id"] = img_id

    image_path = args.image_folder + '/' + img_file
    raw_image = Image.open(image_path).convert("RGB")


    qu = "Please describe this image in detail."
    query = f'<img>{image_path}</img>\n{qu}'



    with torch.inference_mode():
        with torch.no_grad():
            outputs, _ = model.chat(tokenizer, query=query, history=None, temperature=args.temperature, do_sample=False, max_new_tokens=args.max_new_tokens)
            
    print("\n Outputs: ", outputs)

    img_save["caption"] = outputs

    # dump metric file
    
    with open(args.answers_file, "a") as f:
        json.dump(img_save, f)
        f.write('\n')
    
