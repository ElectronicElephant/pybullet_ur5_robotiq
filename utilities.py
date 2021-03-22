import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np


class Models:
    def load_objects(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError


class YCBModels(Models):
    def __init__(self, root, selected_names: tuple = ()):
        self.obj_files = glob.glob(root)
        self.selected_names = selected_names

        self.visual_shapes = []
        self.collision_shapes = []

    def load_objects(self):
        shift = [0, 0, 0]
        mesh_scale = [1, 1, 1]

        for filename in self.obj_files:
            # Check selected_names
            if self.selected_names:
                in_selected = False
                for name in self.selected_names:
                    if name in filename:
                        in_selected = True
                if not in_selected:
                    continue
            print('Loading %s' % filename)
            self.collision_shapes.append(
                p.createCollisionShape(shapeType=p.GEOM_MESH,
                                       fileName=filename,
                                       collisionFramePosition=shift,
                                       meshScale=mesh_scale))
            self.visual_shapes.append(
                p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=filename,
                                    visualFramePosition=shift,
                                    meshScale=mesh_scale))

    def __len__(self):
        return len(self.collision_shapes)

    def __getitem__(self, idx):
        return self.visual_shapes[idx], self.collision_shapes[idx]


class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def rgbd_2_world(self, w, h, d):
        x = (2 * w - self.width) / self.width
        y = -(2 * h - self.height) / self.height
        z = 2 * d - 1
        pix_pos = np.array((x, y, z, 1))
        position = self.tran_pix_world @ pix_pos
        position /= position[3]

        return position[:3]

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb, depth, seg

    def rgbd_2_world_batch(self, depth):
        x = (2 * np.arange(0, self.width) - self.width) / self.width
        x = np.repeat(x[None, :], self.height, axis=0)
        y = -(2 * np.arange(0, self.height) - self.height) / self.height
        y = np.repeat(y[:, None], self.width, axis=1)
        z = 2 * depth - 1

        pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
        position = self.tran_pix_world @ pix_pos.T
        position = position.T
        # print(position)

        position[:, :] /= position[:, 3:4]

        return position[:, :3].reshape(*x.shape, -1)
