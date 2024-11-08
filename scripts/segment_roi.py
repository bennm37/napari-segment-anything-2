import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import sam2
import pickle
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch  # noqa: E402


# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
SAM2_ROOT = os.path.split(os.path.split(sam2.__file__)[0])[0]


def select_device():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with"
            "CUDA and might give numerically different outputs and sometimes"
            " degraded performance on MPS. See e.g. "
            "https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return ax


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle(
            (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
        )
    )


def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def get_frame_names(folder, ext=[".jpg", ".jpeg"]):
    frame_names = [
        p for p in os.listdir(folder) if os.path.splitext(p)[-1] in ext
    ]
    frame_names.sort(key=natural_key)
    return frame_names


def tif_to_jpg(folder):
    frame_names = get_frame_names(folder, ext=[".tif"])
    jpg_folder = f"{folder}/jpgs"
    if not os.path.exists(jpg_folder):
        os.mkdir(jpg_folder)
    for i, frame in enumerate(frame_names):
        img = Image.open(f"{folder}/{frame}")
        jpg_frame = f"{i+1}".zfill(6) + ".jpg"
        img.save(f"{jpg_folder}/{jpg_frame}")


def segment_roi(
    video_dir,
    points,
    labels,
    device,
    checkpoint=f"{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt",
    cfg=f"/{SAM2_ROOT}/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
):
    from sam2.build_sam import build_sam2_video_predictor

    assert os.path.exists(cfg)
    predictor = build_sam2_video_predictor(cfg, checkpoint, device=device)
    tif_to_jpg(video_dir)
    video_dir = f"{video_dir}/jpgs"
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )
    video_segments = (
        {}
    )  # video_segments contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    video_dir = "/Users/nicholb/Documents/data/organoid_data/testcase"
    results_dir = f"{video_dir}/results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    device = select_device()
    points = np.array([[190, 240], [300, 140]], dtype=np.float32)
    labels = np.array([1, 1], np.int32)
    video_segments = segment_roi(video_dir, points, labels, device)
    save(video_segments, f"{results_dir}/video_segments.pkl")
