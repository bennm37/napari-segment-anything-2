from napari import Viewer, run
from napari.plugins import plugin_manager
import numpy as np
import napari_segment_anything_2

plugin_manager.register(napari_segment_anything_2)


def get_prompts(image):
    viewer = Viewer()
    viewer.open(image)
    viewer.window.add_plugin_dock_widget("napari-segment-anything-2")
    viewer.layers.selection = [viewer.layers["SAM points"]]
    centroid = np.array([viewer.layers[0].data.shape]) / 2
    viewer.layers["SAM points"].data = np.concatenate(
        [viewer.layers["SAM points"].data, centroid], axis=0
    )
    viewer.layers["SAM points"].mode = "select"
    viewer.layers["SAM points"].selected_data = [-1]
    run()
    points_data = viewer.layers["SAM points"].data
    return points_data


if __name__ == "__main__":
    image = (
        "/Users/nicholb/Documents/data/organoid_data/testcase/ROI1_stack1.tif"
    )
    prompts = get_prompts(image)
    print(prompts)
