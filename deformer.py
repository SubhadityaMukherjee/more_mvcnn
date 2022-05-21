# %%
from IPython.display import Image
import matplotlib.pyplot as plt
from glob import glob
import pathlib
import os
import math
from open3d import *
import open3d as o3d
import copy
import numpy as np
from utils import *
from tqdm import tqdm
import open3d.visualization.gui as gui
from PIL import Image
# %%


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


# %%

def deform_with_sigma(source, sigma):

    source = "/media/hdd/Datasets/ModelNet40/cup/test/cup_0082.off"
    source = o3d.io.read_triangle_mesh(source)

    source.vertices = normalize3d(source.vertices)
    source.compute_vertex_normals()
    # %%
    all_ims = []
    noises = [0.02, 0.04, 0.06, 0.08, 0.1]
    for noise in tqdm(noises):
        mesh_in = copy.deepcopy(source)
        vertices = np.asarray(mesh_in.vertices)
        vertices += np.random.uniform(0, noise, size=vertices.shape)
        mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_in.compute_vertex_normals()
        R = mesh_in.get_rotation_matrix_from_xyz((-np.pi / 4, 0, np.pi / 4))
        mesh_r = copy.deepcopy(mesh_in)
        mesh_r.rotate(R, center=(0, 0, 0))
        vis = o3d.visualization.Visualizer()
        # works for me with False, on some systems needs to be true
        vis.create_window(visible=False)
        vis.add_geometry(mesh_r)
        # visualizer renders a new frame by calling `poll_events` and `update_renderer`.
        vis.poll_events()
        vis.update_renderer()
        o3d_screenshot_mat = vis.capture_screen_float_buffer()
        # scale and convert to uint8 type
        o3d_screenshot_mat = (
            255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
        # convert to PIL image
        o3d_screenshot_mat = Image.fromarray(o3d_screenshot_mat, "RGB")
        all_ims.append(o3d_screenshot_mat)

# Plot the images of deformed meshes in a grid
# col = 3
# image_count = len(all_ims)
# row = math.ceil(image_count / col)
# plt.figure(figsize=(col * 4, row * 4))
# plt.figure(figsize=(col * 4, row * 4))

# for i, img in enumerate(all_ims):
#     plt.subplot(row, col, i + 1)
#     plt.imshow(img)
#     plt.title("σ = "+str(noises[i]))
#     plt.axis("off")
# plt.savefig("deformed.pdf")
