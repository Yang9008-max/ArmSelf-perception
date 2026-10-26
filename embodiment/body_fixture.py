"""Fixed experimental fixtures and privileged scoring, separate from learners."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from perception.environment import create_static_scene_model
from perception.collect import HOME
from .collect import stereo_scene
from .vision import project


def make_fixture(output, panel=False):
    calibration = stereo_scene('sim/scenes/robot-eye-v6.xml', output/'stereo.xml')
    if panel:
        root = ET.fromstring((output/'stereo.xml').read_text(encoding='utf-8'))
        eye = root.find(".//camera[@name='fixed']")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        setup = mujoco.MjData(model)
        mujoco.mj_forward(model, setup)
        camera_id = model.camera('fixed').id
        rotation = setup.cam_xmat[camera_id].reshape(3, 3)
        optical = np.array([(260-calibration['cx'])*.45/calibration['focal_px'],
                            (100-calibration['cy'])*.45/calibration['focal_px'], .45])
        position = setup.cam_xpos[camera_id]+rotation @ (optical*np.array([1, -1, -1]))
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        ET.SubElement(root.find('worldbody'), 'geom', name='fixed_occlusion_panel', type='box',
                      pos=' '.join(map(str, position)), quat=' '.join(map(str, quaternion)),
                      size='.040 .105 .006', rgba='.28 .37 .44 1', group='3',
                      contype='0', conaffinity='0')
        (output/'stereo.xml').write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')
    model = create_static_scene_model(output/'stereo.xml')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data, nstep=round(.8/model.opt.timestep))
    mujoco.mj_forward(model, data)
    return model, data, calibration


def render_rgb(renderer, data, camera='fixed'):
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    return renderer.render().copy()


def render_truth(renderer, model, data, panel_visible=True):
    option = mujoco.MjvOption()
    option.geomgroup[3] = int(panel_visible)
    renderer.update_scene(data, camera='fixed', scene_option=option)
    renderer.enable_segmentation_rendering()
    labels = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    geom = labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    body = geom & np.isin(labels[..., 0], np.flatnonzero(model.geom_group == 1))
    prop = geom & np.isin(labels[..., 0], np.flatnonzero(model.geom_group == 2))
    panel = geom & np.isin(labels[..., 0], np.flatnonzero(model.geom_group == 3))
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return dict(body_mask=body, prop_mask=prop, panel_mask=panel, depth=depth)


class EvaluationLandmarks:
    """Fixed surface coordinates from home reference rays, NEVER learner inputs.

    A point stays in the denominator even when an algorithm loses it. Anchors
    attach to physical geometry so a different surface point cannot count as
    recovery of the original point merely by lying somewhere on the arm.
    """
    def __init__(self, model, data, pixels, calibration):
        self.calibration = calibration
        camera = model.camera('fixed').id
        rotation = data.cam_xmat[camera].reshape(3, 3)
        position = data.cam_xpos[camera]
        ids, local = [], []
        groups = np.ones(6, np.uint8)
        groups[3] = 0  # The test screen must not redefine the reference body.
        for u, v in np.asarray(pixels):
            direction = rotation @ np.array([(u-calibration['cx'])/calibration['focal_px'],
                                            -(v-calibration['cy'])/calibration['focal_px'], -1.])
            direction /= np.linalg.norm(direction)
            hit = np.array([-1], np.int32)
            distance = mujoco.mj_ray(model, data, position, direction, groups, 1, -1, hit)
            good = distance > 0 and model.geom_group[hit[0]] == 1
            geom = int(hit[0]) if good else -1
            point = position+distance*direction
            ids.append(geom)
            local.append(data.geom_xmat[geom].reshape(3, 3).T @ (point-data.geom_xpos[geom]) if good else np.zeros(3))
        self.ids = np.array(ids)
        self.local = np.array(local)
        self.is_body = self.ids >= 0

    def positions(self, model, data):
        camera = model.camera('fixed').id
        result = np.zeros((len(self.ids), 3), np.float32)
        for i in np.flatnonzero(self.is_body):
            geom = self.ids[i]
            world = data.geom_xpos[geom]+data.geom_xmat[geom].reshape(3, 3) @ self.local[i]
            result[i] = (data.cam_xmat[camera].reshape(3, 3).T @ (world-data.cam_xpos[camera]))*np.array([1, -1, -1])
        return result

    def score_truth(self, model, data, truth):
        xyz = self.positions(model, data)
        uv = project(xyz, self.calibration)
        h, w = truth['depth'].shape
        px = np.rint(uv).astype(int)
        inside = self.is_body & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        px[:, 0] = np.clip(px[:, 0], 0, w-1)
        px[:, 1] = np.clip(px[:, 1], 0, h-1)
        visible = inside & (np.abs(xyz[:, 2]-truth['depth'][px[:, 1], px[:, 0]]) < .015)
        behind_panel = inside & truth['panel_mask'][px[:, 1], px[:, 0]] & ~visible
        return dict(anchor_xyz=xyz, anchor_visible=visible, behind_panel=behind_panel,
                    anchor_body=self.is_body.copy())


def occlusion_protocol(seed):
    rng = np.random.default_rng(seed)
    offset = rng.uniform(-.012, .012, 7)
    offset[[0, 6]] = 0
    yaw = np.r_[np.full(20, .22), np.linspace(.22, -.22, 40),
                np.full(20, -.22), np.linspace(-.22, .22, 40), np.full(20, .22)]
    for i, value in enumerate(yaw):
        target = HOME.copy()+offset
        target[0] = value
        phase = 'clear_start' if i < 20 else 'crossing' if i < 120 else 'clear_return'
        yield phase, target, True
