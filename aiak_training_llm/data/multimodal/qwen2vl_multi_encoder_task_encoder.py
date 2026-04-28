"""Qwen2-VL task encoder with additional SigLIP and DINOv3 pixel streams."""

import math
import re

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision import transforms
from PIL import Image

from megatron.energon import CaptioningSample, VQASample

from aiak_training_llm.data.multimodal import MultiMixQASample
from aiak_training_llm.utils import constants

from .qwen2vl_task_encoder import IGNORE_INDEX, IMAGE_TOKEN_WITH_TAGS, Qwen2VLImageTaskSample, Qwen2VLTaskEncoder


def create_dinov3_processor():
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    size = (512, 512)
    rescale_factor = 1.0 / 255.0

    def preprocess(image):
        # 输入：PIL Image 或 numpy array (H,W,C) uint8 [0,255]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        # 调试：检查 image 类型和转换后的形状
        arr = np.array(image)
        # 1. 转换为 tensor，保持范围 [0,255]，形状 [C, H, W]
        tensor = torch.from_numpy(arr).permute(2,0,1).float()  # [C,H,W], float, 范围 [0,255]
        # 2. Rescale（除以255）
        tensor = tensor * rescale_factor   # 与原始 self.rescale 完全一致
        # 3. Resize（双线性插值，antialias=True）
        tensor = TVF.resize(tensor, size, interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)
        # 4. Normalize
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)
    return preprocess


def create_siglip_processor():
    """返回与 SiglipImageProcessor (TorchvisionBackend) 数值完全一致的预处理函数"""
    # 注意：TorchvisionBackend 内部会 fuse rescale + normalize
    # 即先 resize(uint8)，再用 mean*255, std*255 做 normalize
    # 关键：必须使用 torchvision.transforms.v2.functional (与 transformers 内部一致)
    fused_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) * 255.0
    fused_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) * 255.0
    size = [384, 384]

    def preprocess(image):
        # 输入：PIL Image 或 numpy array (H,W,C) uint8 [0,255]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # 1. PIL -> uint8 tensor [C, H, W]，范围 [0, 255]
        # 与 TorchvisionBackend.process_image 完全一致
        tensor = TVF.pil_to_tensor(image)  # uint8, [C,H,W]

        # 2. Resize（双线性插值，antialias=True）在 uint8 上进行
        # TorchvisionBackend 的 resize 也是直接对 uint8 tensor 操作
        tensor = TVF.resize(
            tensor,
            size,
            interpolation=TVF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # 3. Normalize（fused: 等价于先 /255 再 (x-mean)/std）
        tensor = TVF.normalize(tensor.float(), mean=fused_mean, std=fused_std)
        return tensor.unsqueeze(0)

    return preprocess



class Qwen2VLMultiEncoderTaskEncoder(Qwen2VLTaskEncoder):
    """Extends the default Qwen2-VL encoder with a second SigLIP pixel stream."""

    def __init__(self, args):
        super().__init__(args)
        self.siglip_processor = create_siglip_processor()
        self.dinov3_processor = create_dinov3_processor()

    def _process_with_aux_pixels(self, image, text):
        input_ids, target, pixel_values, image_grid_thw, attn_mask = self._process(image, text)
        siglip_pixel_values = [self.siglip_processor(image)]
        dinov3_pixel_values = [self.dinov3_processor(image)]
        return (
            input_ids,
            target,
            pixel_values,
            siglip_pixel_values,
            dinov3_pixel_values,
            image_grid_thw,
            attn_mask,
        )

    def process_sft_qa(self, messages: list, system: str, raw_video: list, raw_image: list):
        """Process SFT QA data and return both Qwen and SigLIP image pixels."""
        video_grid_thw = None
        pixel_values_videos = []
        image_grid_thw = None
        pixel_values_images = []
        pixel_values_images_siglip = []
        pixel_values_images_dinov3 = []
        video = []
        image = []

        if raw_image is not None:
            for current_image in raw_image:
                pixel_values_images_dinov3.append(self.dinov3_processor(current_image))
                pixel_values_images_siglip.append(self.siglip_processor(current_image))

                resized_image = self._resize_image(current_image)
                image.append(resized_image)

        if raw_video is not None:
            for current_video in raw_video:
                video.append(self._reisize_video(current_video))

        messages, mm_inputs = self.chat_template.mm_plugin.process_messages(
            messages,
            image if image is not None else [],
            video if raw_video is not None else [],
            self.processor,
        )

        if raw_video is not None:
            video_grid_thw = mm_inputs["video_grid_thw"]
            pixel_values_videos = [mm_inputs["pixel_values_videos"]]
        if raw_image is not None:
            image_grid_thw = mm_inputs["image_grid_thw"]
            pixel_values_images = [mm_inputs["pixel_values"]]

        encode_pairs = self.chat_template.encode_multiturn(
            tokenizer=self.tokenizer,
            messages=messages,
            system=system,
        )
        input_ids, target = [], []
        for source_ids, target_ids in encode_pairs:
            input_ids += source_ids + target_ids
            target += [IGNORE_INDEX] * len(source_ids) + target_ids
        input_ids = torch.tensor(input_ids)
        target = torch.tensor(target)
        attn_mask = torch.zeros_like(input_ids).bool()

        return (
            input_ids,
            target,
            attn_mask,
            pixel_values_images,
            pixel_values_images_siglip,
            pixel_values_images_dinov3,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
        )

    def encode_captioning(self, sample: CaptioningSample) -> Qwen2VLImageTaskSample:
        text = IMAGE_TOKEN_WITH_TAGS + sample.caption + self.tokenizer.tokenizer.eos_token
        input_ids, target, imgs, siglip_imgs, dinov3_imgs, image_grid_thw, attn_mask = self._process_with_aux_pixels(
            sample.image, text
        )
        num_tiles = [len(image_grid_thw)]

        input_ids = input_ids[:self.args.seq_length]
        target = target[:self.args.seq_length]
        attn_mask = attn_mask[:self.args.seq_length]

        if self.args.enable_discard_sample and not (target != IGNORE_INDEX).any():
            print(
                f"Discarding sample {sample.__key__} because no valid labels remain after truncation to "
                f"{self.args.seq_length} tokens."
            )
            return None

        if not self.args.enable_discard_sample:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} thw {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            pixel_values_images_siglip=siglip_imgs,
            pixel_values_images_dinov3=dinov3_imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_vqa4packing(self, sample: VQASample) -> Qwen2VLImageTaskSample:
        text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": sample.context},
                {"role": "assistant", "content": sample.answers},
            ],
            tokenize=False,
        ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)

        if text[-1] == "\n":
            text = text[:-1]

        input_ids, _, imgs, siglip_imgs, dinov3_imgs, image_grid_thw, attn_mask = self._process_with_aux_pixels(
            sample.image, text
        )
        target = torch.ones_like(input_ids) * IGNORE_INDEX
        answers = self.tokenizer.tokenize(sample.answers)
        target[-len(answers) - 1 : -1] = torch.tensor(answers)
        target[-1] = input_ids[-1]

        num_tiles = [len(image_grid_thw)]
        input_ids = input_ids[: self.args.seq_length]
        target = target[: self.args.seq_length]
        attn_mask = attn_mask[: self.args.seq_length]

        if self.args.enable_discard_sample and not (target != IGNORE_INDEX).any():
            print(
                f"Discarding sample {sample.__key__} because no valid labels remain after truncation to "
                f"{self.args.seq_length} tokens."
            )
            return None

        if not self.args.enable_discard_sample:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            pixel_values_images_siglip=siglip_imgs,
            pixel_values_images_dinov3=dinov3_imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_multi_mix_qa(self, sample: MultiMixQASample) -> Qwen2VLImageTaskSample:
        try:
            if self.args.training_phase == constants.TrainingPhase.SFT:
                num_tiles = []
                (
                    input_ids,
                    target,
                    attn_mask,
                    imgs,
                    siglip_imgs,
                    dinov3_imgs,
                    image_grid_thw,
                    pixel_values_videos,
                    video_grid_thw,
                ) = self.process_sft_qa(sample.messages, sample.system, sample.video, sample.image)
                if sample.video is not None:
                    num_tiles = [len(video_grid_thw)]
                elif sample.image is not None:
                    num_tiles = [len(image_grid_thw)]
            else:
                raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")
        except ValueError as exc:
            print(f"Skipping sample {sample.__key__} due to data inconsistency: {exc}")
            return None

        if len(input_ids) == 0:
            print(f"Skipping sample {sample.__key__} because input_ids is empty after processing.")
            return None

        input_ids = input_ids[: self.args.seq_length]
        target = target[: self.args.seq_length]
        attn_mask = attn_mask[: self.args.seq_length]

        if self.args.enable_discard_sample and not (target != IGNORE_INDEX).any():
            print(
                f"Discarding sample {sample.__key__} because no valid labels remain after truncation to "
                f"{self.args.seq_length} tokens."
            )
            return None

        if not self.args.enable_discard_sample:
            if sample.video is not None:
                assert video_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, (
                    f"{sample.__key__} grid_thw: {video_grid_thw}"
                )
            elif sample.image is not None:
                assert image_grid_thw.prod(dim=-1).sum() / 4 <= self.args.seq_length, (
                    f"{sample.__key__} grid_thw: {image_grid_thw}"
                )

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            pixel_values_images_siglip=siglip_imgs,
            pixel_values_images_dinov3=dinov3_imgs,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    def encode_vaq(self, sample: VQASample) -> Qwen2VLImageTaskSample:
        if self.args.training_phase == constants.TrainingPhase.PRETRAIN:
            if self.args.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, siglip_imgs, dinov3_imgs, image_grid_thw, attn_mask = self._process_with_aux_pixels(
                sample.image, text
            )
        elif self.args.training_phase == constants.TrainingPhase.SFT:
            if len(sample.answers) < 1:
                raise ValueError("sample.answers < 1!")

            if sample.image is not None:
                img_arr = np.array(sample.image)
                if np.sum(img_arr) == 0:
                    raise ValueError("Image pixels are all zero!")

            max_answer_length = self.args.training_rice_vl_max_answer_length
            if len(sample.answers) > max_answer_length:
                original_length = len(sample.answers)
                preliminary_cut = sample.answers[:max_answer_length]
                cleaned_cut = preliminary_cut.rstrip(".。 \t\n")
                matches = list(re.finditer(r"[.。]", cleaned_cut))
                sample.answers = cleaned_cut[: matches[-1].end()] if matches else preliminary_cut
                print(
                    "Answer truncated to a full sentence. "
                    f"Original length: {original_length}, New length: {len(sample.answers)}"
                )

            text = self.processor.apply_chat_template(
                [
                    {"role": "user", "content": sample.context},
                    {"role": "assistant", "content": sample.answers},
                ],
                tokenize=False,
            ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            if text[-1] == "\n":
                text = text[:-1]
            input_ids, _, imgs, siglip_imgs, dinov3_imgs, image_grid_thw, attn_mask = self._process_with_aux_pixels(
                sample.image, text
            )
            target = torch.ones_like(input_ids) * IGNORE_INDEX
            answers = self.tokenizer.tokenize(sample.answers)
            target[-len(answers) - 1 : -1] = torch.tensor(answers)
            target[-1] = input_ids[-1]
        else:
            raise NotImplementedError(f"Unknown training phase {self.args.training_phase}")

        num_tiles = [len(image_grid_thw)]
        input_ids = input_ids[: self.args.seq_length]
        target = target[: self.args.seq_length]
        attn_mask = attn_mask[: self.args.seq_length]

        if self.args.enable_discard_sample and not (target != IGNORE_INDEX).any():
            print(
                f"Discarding sample {sample.__key__} because no valid labels remain after truncation to "
                f"{self.args.seq_length} tokens."
            )
            return None

        if not self.args.enable_discard_sample:
            assert image_grid_thw.prod() / 4 <= self.args.seq_length, f"{sample.__key__} grid_thw: {image_grid_thw}"

        return Qwen2VLImageTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            pixel_values_images_siglip=siglip_imgs,
            pixel_values_images_dinov3=dinov3_imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )
