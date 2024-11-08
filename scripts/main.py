import os
import numpy as np
import matplotlib.pyplot as plt
from segment_roi import select_device, segment_roi, save, tif_to_jpg
from show_masks import animate, get_images, get_masks
from get_prompts import get_prompts

if __name__ == "__main__":
    video_dir = "/Users/nicholb/Documents/data/organoid_data/seperatedStacks/image_1_MMStack_control_DMSO_1-3.ome_restacked/ROI3"  # noqa E501
    # video_dir = "/Users/nicholb/Documents/data/organoid_data/testcase_small"
    tif_to_jpg(video_dir)
    results_dir = f"{video_dir}/results"
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    device = select_device()
    points = get_prompts(f"{video_dir}/ROI3_stack1.tif")
    # save points
    np.savetxt(
        f"{results_dir}/points.csv", points, delimiter=",", header="x,y"
    )
    labels = np.array([1 for _ in range(points.shape[0])], np.int32)
    video_segments = segment_roi(video_dir, points, labels, device)
    save(video_segments, f"{results_dir}/video_segments.pkl")
    masks = get_masks(video_dir)
    images = get_images(video_dir)
    print(f"Found {len(masks)} masks and {len(images)} images")
    anim, fig, ax = animate(images, masks)
    plt.show()
    
