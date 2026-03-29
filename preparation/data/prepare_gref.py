import numpy as np
import os

from typing import Dict, List, Union, Tuple
from grefer import G_REFER
from pycocotools import mask as maskUtils
from datasets import Dataset, Features, Value, Sequence, Array2D, Array3D, Image
from pathlib import Path
from PIL import Image as PILImage
from segment_anything.utils.transforms import ResizeLongestSide
from gen_sam_info import resize_pad_mask


def coco_anno_to_mask(raw_anno: Dict, orig_size: Union[Tuple, List]) -> np.ndarray:
    rles = maskUtils.frPyObjects(raw_anno["segmentation"], orig_size[0], orig_size[1])
    mask = maskUtils.decode(rles)
    if mask.ndim == 3:  # Union
        mask = np.any(mask, axis=2)
    mask_bool = mask.astype(np.bool_)
    return mask_bool


class GrefLoader:
    def __init__(self, anno_root, image_dir, split, split_by="unc"):
        refer_api = G_REFER(data_root=anno_root, image_dir=image_dir, dataset="grefcoco", splitBy=split_by)
        ref_ids = refer_api.getRefIds(split=split)
        refs = refer_api.loadRefs(ref_ids)
        imgs = [refer_api.loadImgs(ref['image_id'])[0] for ref in refs]
        anns = [refer_api.loadAnns(ref['ann_id']) for ref in refs]
        self.split = split
        self.imgs_refs_anns = list(zip(imgs, refs, anns))

    def load_item(self, idx):
        img_dict, ref_dict, anno_dicts = self.imgs_refs_anns[idx]
        image_id = int(img_dict["id"])
        image_filename = img_dict["file_name"]
        original_hw = [int(img_dict["height"]), int(img_dict["width"])]
        text = [s["raw"] for s in ref_dict["sentences"]]

        # Assertion for safety
        assert ref_dict['image_id'] == image_id
        assert ref_dict['split'] == self.split
        ann_id = ref_dict['ann_id'] if isinstance(ref_dict['ann_id'], list) else [ref_dict['ann_id']]

        # No target samples
        if None in anno_dicts:
            assert anno_dicts == [None]
            assert ann_id == [-1]
            mask = np.zeros((0, original_hw[0], original_hw[1]), dtype=bool)
            bbox = np.zeros((0, 4), dtype=np.float32)
        else:
            mask_list = [coco_anno_to_mask(anno, original_hw) for anno in anno_dicts]
            mask = np.stack(mask_list, axis=0)
            bbox = np.asarray([anno["bbox"] for anno in anno_dicts], dtype=np.float32)

        return {
            # "image": image,
            "image_filename": image_filename,
            "text": text,
            "mask": mask.tolist(),
            "image_id": image_id,
            "ann_id": ann_id,
            "bbox": bbox.tolist(),
            "original_hw": original_hw,
            # "unpadded_hw": [int(x) for x in input_size],
            # "mask_float_256": mask_float_256,
            # "embed_path": embed_path.
        }

    def get_length(self):
        return len(self.imgs_refs_anns)


def save_gref_hf(
        anno_root, image_dir, split, save_root,
        max_num_items=2000, st=None, ed=None
):
    """
    The function refers to that in gen_sam_info.py
    """
    features = Features({
        "image": Image(decode=True),
        "image_filename": Value("string"),
        "text": Sequence(Value("string")),
        "mask": Sequence(Sequence(Sequence(Value("bool")))),
        "image_id": Value("int64"),
        "ann_id": Sequence(Value("int64")),
        "bbox": Sequence(Sequence(Value("float32"), length=4)),
        "original_hw": Sequence(Value("int32"), length=2),
        "unpadded_hw": Sequence(Value("int32"), length=2),
        "mask_float_256":  Sequence(Array2D(shape=(256, 256), dtype="float16")),
        "embed_path": Value("string"),
    })

    gref_loader = GrefLoader(anno_root, image_dir, split)
    whole_len = gref_loader.get_length()
    start = 0 if st is None else st
    end = whole_len if ed is None else ed
    data_buffers = []

    i_start = start
    for i in range(start, end):
        data_item = gref_loader.load_item(i)
        image = PILImage.open(os.path.join(image_dir, data_item["image_filename"])).convert("RGB")
        image_np = np.array(image)
        orig_size = data_item["original_hw"]
        if not (orig_size[0] == image_np.shape[0] and orig_size[1] == image_np.shape[1]):
            print("Warning! original size not matched", flush=True)

        resize = ResizeLongestSide(target_length=1024)
        orig_size = (image_np.shape[0], image_np.shape[1])
        input_size = resize.get_preprocess_shape(orig_size[0], orig_size[1], resize.target_length)

        embed_name = data_item["image_filename"].rsplit('.', 1)[0] + ".pt"
        embed_path = os.path.join("b" + str(data_item["image_id"])[-1], embed_name)
        if len(data_item["mask"]) == 0:
            mask_float_256 = []
        else:
            m = resize_pad_mask(np.array(data_item["mask"], dtype=np.float32), target_len=256)
            m = m.cpu().numpy().astype(np.float16, copy=False)
            mask_float_256 = [np.ascontiguousarray(m[i]) for i in range(m.shape[0])]

        data_item["image"] = image
        data_item["unpadded_hw"] = [int(x) for x in input_size]
        data_item["embed_path"] = embed_path
        data_item["mask_float_256"] = mask_float_256
        data_buffers.append(data_item)

        if i % 100 == 0:
            print_split = split if split else "whole"
            print(f"[{i}/{whole_len}] [{print_split}] Examples processed...Start: {start}...End: {end}", flush=True)

        if len(data_buffers) >= max_num_items and end - start != max_num_items:
            save_dir = os.path.join(save_root, f"{i_start:05d}_{i + 1:05d}")
            i_start = i + 1
            dataset = Dataset.from_list(data_buffers, features=features)
            dataset.save_to_disk(save_dir)
            print(f"buffers saved: {save_dir}", flush=True)
            data_buffers = []
            del dataset

    # At the end, save the latest buffer or save without buffering
    save_dir = save_root if end - start <= max_num_items else os.path.join(save_root, f"{i_start:05d}_{end:05d}")
    if data_buffers:
        dataset = Dataset.from_list(data_buffers, features=features)
        dataset.save_to_disk(save_dir)
        print(f"saved: {save_dir}", flush=True)


if __name__ == "__main__":
    save_gref_hf(
        anno_root="raw_data",
        image_dir="raw_data/coco_train2014",
        split="train",
        save_root="data/grefcoco/grefcoco_train_slices",
        st=60000, ed=None
    )
