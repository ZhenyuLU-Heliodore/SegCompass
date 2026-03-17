import os
import numpy as np
import glob
import torch
import verl.utils.torch_functional as verl_F
import json
import re

from pathlib import Path
from typing import Optional, Dict, List
from datasets import load_from_disk, DatasetDict
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image as PILImage
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.rl_dataset import process_image
from verl.models.transformers.qwen2_5_vl import get_rope_index
from PIL import Image


class SaeCacheActDataset(Dataset):
    def __init__(
            self,
            img_txt_ds: Dataset,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            llm_version: str,
            max_prompt_length: int = 1400,
            max_pixels: int = 705600,
            min_pixels: int = 3136,
            system_prompt: Optional[str] = None,
    ):
        super().__init__()
        self.img_txt_ds = img_txt_ds
        self.tokenizer = tokenizer
        self.processor = processor
        self.llm_version = llm_version
        self.max_prompt_length = max_prompt_length
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.system_prompt = r"You are a helpful assistant." if system_prompt is None else system_prompt
        self.user_prompt = "<image> {text}"

    def __len__(self):
        return len(self.img_txt_ds)

    def _arrange_qwen_tokens(self, row_dict):
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt.format(text=row_dict["text"])}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if "images" not in row_dict:
            raw_prompt, image_grid_thw = prompt, None
            image_inputs = {"pixel_values": None, "image_grid_thw": None}
        else:
            raw_prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            image_inputs = self.processor.image_processor(row_dict["images"], return_tensors="pt")
            image_grid_thw = image_inputs["image_grid_thw"]

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size ** 2
                index = 0
                while "<image>" in prompt:
                    num_placeholder = int((image_grid_thw[index].prod() // merge_length).item())
                    prompt = prompt.replace(
                        "<image>", "<|vision_start|>" + "<|placeholder|>" * num_placeholder + "<|vision_end|>", 1,
                    )
                    index += 1
                prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

        row_dict.update(image_inputs)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )
        if "images" in row_dict:
            position_ids = get_rope_index(
                self.processor, input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        return input_ids, attention_mask, position_ids, raw_prompt

    def _arrange_llava_tokens(self, row_dict):
        vl_messages = [{"role": "system", "content": self.system_prompt},
                       {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row_dict["text"]},]},]
        text_messages = [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": self.user_prompt.format(text=row_dict["text"])}]

        text_prompt = self.processor.apply_chat_template(text_messages, add_generation_prompt=True, tokenize=False)
        raw_prompt = text_prompt
        if "images" in row_dict:
            prompt = self.processor.apply_chat_template(vl_messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = text_prompt

        assert self.processor.tokenizer.padding_side == "left"  # must be left padding here
        proc = self.processor(
            text=[prompt], images=row_dict["images"] if "images" in row_dict else None, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_prompt_length,
        )
        if "images" in row_dict:
            image_inputs = {"pixel_values": proc["pixel_values"], "image_grid_thw": None}
        else:
            image_inputs = {"pixel_values": None, "image_grid_thw": None}
        row_dict.update(image_inputs)

        return (
            proc["input_ids"][0],
            proc["attention_mask"][0],
            torch.clip(proc["attention_mask"][0].cumsum(dim=0) - 1, min=0, max=None),
            raw_prompt
        )

    def __getitem__(self, idx: int) -> Dict:
        row_dict: Dict = self.img_txt_ds[idx]  # image, text image_filename
        if "image" in row_dict:
            row_dict["images"] = [row_dict["image"]]
        if "images" in row_dict:  # expand image token
            row_dict["images"] = [process_image(img, self.max_pixels, self.min_pixels) for img in row_dict["images"]]

        if self.llm_version in ["qwen-2.5"]:
            input_ids, attention_mask, position_ids, raw_prompt = self._arrange_qwen_tokens(row_dict)
        elif self.llm_version in ["llava-1.5"]:
            input_ids, attention_mask, position_ids, raw_prompt = self._arrange_llava_tokens(row_dict)
        else:
            raise ValueError("Wrong type of LLM!")

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        return row_dict


class HfImgTxtDataset(Dataset):
    def __init__(self, data_dir: str, image_dir: Optional[str] = None, split: Optional[str] = None):
        super().__init__()
        ds = load_from_disk(data_dir)
        self.ds = ds[split] if isinstance(ds, DatasetDict) else ds
        self.image_dir = image_dir
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        item: Dict = self.ds[idx]
        image = self._get_image(item)
        text = self._get_text(item)
        image_filename = item["image_filename"]
        hidden_path = _get_hidden_path(image_filename, self.data_dir)
        return {"image": image, "text": text, "image_filename": image_filename, "hidden_path": hidden_path}

    def _get_text(self, item: Dict) -> str:
        prob_key = "text" if "text" in item else ("problem" if "problem" in item else "prompt")
        text = item[prob_key]
        if isinstance(text, list):
            chosen = next((s for s in text if isinstance(s, str) and s.strip()), "")
            return chosen
        return str(text) if text else ""

    def _get_image(self, item: Dict) -> PILImage.Image:
        if self.image_dir is not None and "image_filename" in item:
            path = os.path.join(self.image_dir, item["image_filename"])
            with Image.open(path) as im:
                image = im.convert("RGB")
            return image
        else:
            image = item["image"]

        if isinstance(image, PILImage.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return PILImage.fromarray(image).convert("RGB")

        raise TypeError(f"Wrong image type：{type(image)}")


class JsonlImgTxtDataset(Dataset):
    def __init__(self, data_dir: str, image_dir: str = None, split: str = None, num_shards: int = 8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_dir = image_dir  # placeholder for consistency
        self.split = split          # placeholder for consistency

        self._shards, self._items, self._idx2shard, self._idx2local = [], [], [], []
        for i in range(num_shards):
            p = self.data_dir / f"part_{i}.jsonl"
            if not p.exists():
                raise FileNotFoundError(f"Missing shard file part_{i}.jsonl in {self.data_dir}")
            self._shards.append(p)

        # global idx -> (shard_id, local_idx)
        for shard_id, p in enumerate(self._shards):
            local_id = 0
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    self._items.append(json.loads(line))
                    self._idx2shard.append(shard_id)
                    self._idx2local.append(local_id)
                    local_id += 1

        self._length = len(self._items)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict:
        item = self._items[idx]
        phrases = item["conversation"][0]["phrases"]
        image, text, image_filename = self._get_image_text(phrases)
        if text == "":
            print(f"Warning: the sample {image_filename} with empty text!")
        hidden_path = _get_hidden_path(image_filename, self.data_dir)

        return {"image": image, "text": text, "image_filename": image_filename, "hidden_path": hidden_path}

    def _get_image_text(self, phrases):
        # get image, text, image_filename
        for p in phrases:
            img_dict: Optional[Dict] = p["image"] if "image" in p else None
            if isinstance(img_dict, dict):
                image_path = img_dict.get("img_path", None) or img_dict.get("ori_image", None)
                if not image_path:
                    continue
                try:
                    with Image.open(image_path) as im:
                        image = im.convert("RGB")
                    text = img_dict.get("after_text", "")
                    image_filename = Path(image_path).name
                    return image, text, image_filename
                except Exception:
                    continue

        return Image.new("RGB", (256, 256), (255, 255, 255)), "", "blank_zzzz.jpg"


def _get_hidden_path(image_filename, dataset_path):
    image_stem = Path(image_filename).stem
    last2 = image_stem[-2:].lower() if len(image_stem) >= 2 else ""
    bucket = f"b{last2}" if re.compile(r"^[0-9a-f]{2}$").match(last2) else "others"

    dataset_name = re.match(r"[^_]*_", Path(dataset_path).name)
    prefix = dataset_name.group(0) if dataset_name else ""

    return str(Path(bucket) / f"{prefix}{image_stem}.npz")


def is_hf_dataset_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_info_here = os.path.isfile(os.path.join(path, "dataset_info.json"))
    has_info_in_child = any(
        os.path.isfile(os.path.join(path, d, "dataset_info.json"))
        for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )
    has_arrow = bool(glob.glob(os.path.join(path, "**", "*.arrow"), recursive=True))
    if (has_info_here or has_info_in_child) and has_arrow:
        return True
    try:
        _ = load_from_disk(path)
        return True
    except Exception:
        return False


def build_mixed_dataset(
        data_dirs: List[str],
        image_dirs: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
) -> Dataset:
    def _norm_none(x):
        return None if (isinstance(x, str) and x.strip().lower() in {"none", ""}) else x

    n = len(data_dirs)
    image_dirs = [None] * n if image_dirs is None else [_norm_none(p) for p in image_dirs]
    splits = [None] * n if splits is None else [_norm_none(s) for s in splits]
    assert n == len(image_dirs) == len(splits)

    ds_list = []
    for data_dir, image_dir, split in zip(data_dirs, image_dirs, splits):
        if is_hf_dataset_dir(data_dir):
            dataset = HfImgTxtDataset(data_dir, image_dir, split)
        elif os.path.isdir(data_dir):
            dataset = JsonlImgTxtDataset(data_dir, image_dir, split)
        else:
            raise ValueError(f"Invalid data dir: {data_dir}")
        ds_list.append(dataset)

    return ConcatDataset(ds_list)

