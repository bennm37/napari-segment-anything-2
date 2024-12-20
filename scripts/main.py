import os
import numpy as np
from napari import run
from segment_roi import select_device, segment_roi, save, tif_to_jpg, get_frame_names
from show_masks import show_masks
from get_prompts import get_prompts

if __name__ == "__main__":
    # video_dir = f"{organoid_data_dir}/seperatedStacks/image_1_MMStack_control_DMSO_1-3.ome_restacked/ROI3"
    video_dirs = [
        "/Users/nicholb/Documents/data/organoid_data/seperatedStacks/image_1_MMStack_control_DMSO_1-3.ome_restacked/ROI3",
        "/Users/nicholb/Documents/data/organoid_data/testcase_small",
    ]
    # labelling
    for video_dir in video_dirs:
        jpg_folder = tif_to_jpg(video_dir)
        jpgs = get_frame_names(jpg_folder)
        results_dir = f"{video_dir}/results"
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        device = select_device()
        points = get_prompts(f"{jpg_folder}/{jpgs[0]}")
        # save points
        np.savetxt(f"{results_dir}/points.csv", points, delimiter=",", header="x,y")

    # segmenting : this should be done on gpu
    for video_dir in video_dirs:
        restults_dir = f"{video_dir}/results"
        points = np.loadtxt(f"{results_dir}/points.csv", delimiter=",", skiprows=1).reshape(-1, 2)
        labels = np.array([1 for _ in range(points.shape[0])], np.int32)
        video_segments = segment_roi(video_dir, points, labels, device)
        save(video_segments, f"{results_dir}/video_segments.pkl")
        show_masks(video_dir)
        run()
