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

from memvr import apply_memvr_glm

# model_path = "../../LLaVA/llava-v1.5-7b"
model_path = "../../LLaVA/glm-4v-9b"

device = "cuda:0"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map=device) 
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device,
    local_files_only=True,
    cache_dir=model_path,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)


image_name = '../MemVR/images/case_1.png'
image = Image.open(image_name).convert("RGB")


question = "What is the name of this famous sight in the photo?"

inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": question}],
                                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                    return_dict=True, device_map=device)
inputs = inputs.to(device)
gen_kwargs = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 0,
    }

# Try disable / enable it to see the difference
apply_memvr_glm(
    self=model,
    starting_layer = 5,    
    ending_layer = 16,
    entropy_threshold = 0.75,
    retracing_ratio = 0.1
)

with torch.inference_mode():
    start_time = time.time()

    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]    
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()

print(outputs)
print(f"Time taken: {end_time - start_time} seconds")
