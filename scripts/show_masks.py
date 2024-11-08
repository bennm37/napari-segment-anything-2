from segment_roi import show_mask, load, get_frame_names
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np
from napari import Viewer, run

CYAN = {0: [0, 0, 0, 0], 1: [0, 0.8, 0.8, 0.5]}


def animate(images, masks):
    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        ax.imshow(images[i], cmap="gray")
        _ = show_mask(masks[i], ax)

    anim = FuncAnimation(
        fig, update, frames=len(images), interval=30, repeat=False
    )
    return anim, fig, ax


def get_images(video_dir):
    jpg_folder = f"{video_dir}/jpgs"
    frame_names = get_frame_names(jpg_folder)
    images = [
        np.array(Image.open(f"{jpg_folder}/{frame}")) for frame in frame_names
    ]
    return images


def get_masks(filename):
    video_segments = load(filename)
    masks = [video_segments[i][1] for i in range(len(video_segments))]
    return masks


def show_masks(video_dir, name="video_segments.pkl"):
    viewer = Viewer()
    viewer.open(video_dir)
    masks = get_masks(f"{video_dir}/results/{name}")
    viewer.add_labels(np.array(masks)[:, 0], name="SAM masks", colormap=CYAN)
    return viewer


if __name__ == "__main__":
    video_dir = "/Users/nicholb/Documents/data/organoid_data/seperatedStacks/image_1_MMStack_control_DMSO_1-3.ome_restacked/ROI3"  # noqa E501
    viewer = show_masks(video_dir)
    run()
