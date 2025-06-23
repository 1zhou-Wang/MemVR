from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from llava.model.llava_arch import LlavaMetaForCausalLM

from PIL import Image
import requests
import copy
import torch
import time

from memvr import apply_memvr_qwen

# model_path = "../../LLaVA/llava-v1.5-7b"
model_path = "./Qwen-VL-Chat"

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=device, trust_remote_code=True).eval()

image_name = '../MemVR/images/case_1.png'
question = "What is the name of this famous sight in the photo?"

# Try disable / enable it to see the difference
apply_memvr_qwen(
    self=model,
    starting_layer = 9,    
    ending_layer = 16,
    entropy_threshold = 0.75,
    retracing_ratio = 0.1
)

with torch.inference_mode():
    start_time = time.time()

    query = f'<img>{image_name}</img>\n{question}'
    outputs, _ = model.chat(tokenizer, query=query, history=None, temperature=0, do_sample=False, max_new_tokens=128)

    end_time = time.time()

print(outputs)
print(f"Time taken: {end_time - start_time} seconds")
