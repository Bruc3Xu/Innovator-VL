"""Qwen2-VL task encoder with additional SigLIP and DINOv3 pixel streams."""

import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from megatron.energon import CaptioningSample, VQASample
from transformers import AutoProcessor

from aiak_training_llm.data.multimodal import MultiMixQASample
from aiak_training_llm.utils import constants

from .qwen2vl_task_encoder import IGNORE_INDEX, IMAGE_TOKEN_WITH_TAGS, Qwen2VLImageTaskSample, Qwen2VLTaskEncoder


class Qwen2VLMultiEncoderTaskEncoder(Qwen2VLTaskEncoder):
    """Extends the default Qwen2-VL encoder with a second SigLIP pixel stream."""

    def __init__(self, args):
        super().__init__(args)
        model_path = "/mnt/si00068187c7/default/innovator_vl/models/"
        self.siglip_processor = AutoProcessor.from_pretrained(model_path + "siglip2-so400m-patch14-384", use_fast=True)
        self.dinov3_processor = AutoProcessor.from_pretrained(model_path + "dinov3-vitl16-pretrain-lvd1689m", use_fast=True, size=(448, 448))

    def _to_image_list(self, images):
        if images is None:
            return []
        if isinstance(images, list):
            return images
        return [images]

    def _process_siglip_images(self, images):
        image_list = self._to_image_list(images)
        if len(image_list) == 0:
            return []
        pixel_values = self.siglip_processor(images=image_list, return_tensors="pt")["pixel_values"]
        return [pixel_values]

    def _process_dinov3_images(self, images):
        image_list = self._to_image_list(images)
        if len(image_list) == 0:
            return []
        pixel_values = self.dinov3_processor(images=image_list, return_tensors="pt")["pixel_values"]

    def _process_with_aux_pixels(self, image, text):
        input_ids, target, pixel_values, image_grid_thw, attn_mask = self._process(image, text)
        siglip_pixel_values = self._process_siglip_images(image)
        dinov3_pixel_values = self._process_dinov3_images(image)
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
                resized_image = self._resize_image(current_image)
                image.append(resized_image)
            pixel_values_images_siglip = self._process_siglip_images(image)

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
            pixel_values_images_dinov3 = self._process_dinov3_images_from_qwen(
                mm_inputs["pixel_values"],
                image_grid_thw,
            )

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
