from napari import run
import numpy as np
from show_masks import show_masks
from segment_roi import segment_roi, save, select_device
import os


def corrections(video_dir, name="video_segments.pkl"):
    viewer = show_masks(video_dir, name)
    if os.path.exists(f"{video_dir}/results/corrected_points.csv"):
        points = np.loadtxt(f"{video_dir}/results/corrected_points.csv", delimiter=",")
        positive_points = points[points[:, -1].astype(int) == 1, :-1]
        negative_points = points[points[:, -1].astype(int) == 0, :-1]
    elif os.path.exists(f"{video_dir}/results/points.csv"):
        positive_points = np.loadtxt(f"{video_dir}/results/points.csv", delimiter=",")
        positive_points = positive_points.reshape(-1, 2)
        positive_points = np.hstack([np.zeros((positive_points.shape[0], 1)), positive_points])
        negative_points = np.empty((0, 3))
    else:
        print(f"Warning: No points found at {video_dir}/results/points.csv")
        positive_points = np.empty((0, 3))
        negative_points = np.empty((0, 3))

    viewer.add_points(positive_points, name="Positive points", face_color="green")
    viewer.add_points(negative_points, name="Negative points", face_color="red")
    viewer.layers["Negative points"].mode = "add"
    viewer.layers["Positive points"].mode = "add"
    run()
    return (
        viewer.layers["Positive points"].data,
        viewer.layers["Negative points"].data,
    )


def create_labels(positive, negative):
    labels = np.zeros(positive.shape[0] + negative.shape[0])
    labels[: positive.shape[0]] = 1
    points = np.vstack([positive, negative])
    return points, labels.astype(int)


if __name__ == "__main__":
    video_dir = "/Users/nicholb/Documents/data/organoid_data/testcase_small"
    # video_dir = "/Users/nicholb/Documents/data/organoid_data/seperatedStacks/image_1_MMStack_control_DMSO_1-3.ome_restacked/ROI3"
    if os.path.exists(f"{video_dir}/results/video_segments_corrected.pkl"):
        positive, negative = corrections(video_dir, "video_segments_corrected.pkl")
    else:
        positive, negative = corrections(video_dir, "video_segments.pkl")
    points, labels = create_labels(positive, negative)
    np.savetxt(
        f"{video_dir}/results/corrected_points.csv",
        np.hstack([points, labels[:, None]]),
        delimiter=",",
        header="t,x,y,label",
    )
    video_segments = segment_roi(video_dir, points, labels, select_device())
    save(video_segments, f"{video_dir}/results/video_segments_corrected.pkl")
    _, _ = corrections(video_dir, "video_segments_corrected.pkl")
    run()
