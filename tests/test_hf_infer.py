import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = (
    "/mnt/si00068187c7/default/innovator_vl/Innovator-VL/checkpoints/instruct_release"
)

# model_path = "/mnt/si000268ks12/default/wuyanfeng/models/Innovator-VL-8B-Instruct/"

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/mnt/si00068187c7/default/innovator_vl/Innovator-VL/tests/733_1.jpg",
            },
            {
                "type": "image",
                "image": "/mnt/si00068187c7/default/innovator_vl/Innovator-VL/tests/733_2.jpg",
            },
            {
                "type": "text",
                "text": "Question: <image 1> What group of pathogens, often mistaken for regrowth following glyphosate treatment, can cause a growth habit in blackberry plants that is near-identical to the 'little leaf' symptoms commonly witnessed post-glyphosate treatment?\nOptions:\nA. I don't know and I don't want to guess\nB. Nematodes\nC. Fungi\nD. Phytoplasmas\nE. Bacteria\nPlease select the correct answer from the options above.",
            },
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
# print(inputs.pixel_values.shape, inputs.pixel_values_images_siglip.shape, inputs.pixel_values_images_dinov3.shape)

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="cuda", trust_remote_code=True
)
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
