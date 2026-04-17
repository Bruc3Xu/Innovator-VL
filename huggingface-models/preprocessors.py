import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SiglipVisionModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)


def dinov3_fwd():
    pretrained_model_name = "models/dinov3-vitl16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(pretrained_model_name).to("cuda:0")
    print(model)
    patch_size = model.config.patch_size
    print("Patch size:", patch_size) # 16
    print("Num register tokens:", model.config.num_register_tokens) # 4

    inputs = processor(images=image, return_tensors="pt")
    print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

    batch_size, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
    num_patches_flat = num_patches_height * num_patches_width

    with torch.inference_mode():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
    assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

    cls_token = last_hidden_states[:, 0, :]
    patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
    patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
    print(patch_features.shape)
    return patch_features_flat


def siglip_fwd():
    model = SiglipVisionModel.from_pretrained("models/siglip2-so400m-patch14-384")
    print(model)
    processor = AutoProcessor.from_pretrained("models/siglip2-so400m-patch14-384")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    # pooled_output = outputs.pooler_output  # pooled features
    return last_hidden_state


# dinov3_fwd()
siglip_fwd()