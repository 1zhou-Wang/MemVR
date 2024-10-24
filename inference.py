from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llava.model.llava_arch import LlavaMetaForCausalLM

from PIL import Image
import requests
import copy
import torch
import time

from memvr import apply_memvr_llama

model_path = "llava-v1.5-7b"

device = "cuda:0"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path = model_path,
    model_base = None, 
    model_name = get_model_name_from_path(model_path), 
    device_map = device 
    ) # Add any other thing you want to pass in llava_model_args

model.eval()
model.tie_weights()
image_name = 'images/case_1.png'
image = Image.open(image_name).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "vicuna_v1" # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nWhat is the name of this famous sight in the photo?"
conv_mode = "vicuna_v1"
conv = conv_templates[conv_mode].copy()
inp =  question
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
question = conv.get_prompt()
    
input_ids = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

# Try disable / enable it to see the difference
apply_memvr_llama(
    self=model,
    starting_layer = 5,    
    ending_layer = 16,
    entropy_threshold = 0.75,
    retracing_ratio = 0.12
)

with torch.inference_mode():
    start_time = time.time()
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=1,
        max_new_tokens=1024,
    )
    end_time = time.time()
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print(text_outputs)
print(f"Time taken: {end_time - start_time} seconds")