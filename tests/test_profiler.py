from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


model_path = "/mnt/si00068187c7/default/innovator_vl/Innovator-VL/checkpoints/instruct_release"
model_path = "/mnt/si000268ks12/default/wuyanfeng/models/Innovator-VL-8B-Instruct/"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", trust_remote_code=True
)


# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image with detailed info."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
    on_trace_ready=tensorboard_trace_handler("./log_base/"),
    record_shapes=True,
    with_stack=True
) as prof:
    for _ in range(3):
        output = model.generate(**inputs, max_new_tokens=128)
        prof.step()

