"""Procedural noncontact MuJoCo distractors, independent of robot commands.

Geometry identities are used only by scene setup and evaluation rendering. The
learning pipeline receives rendered RGB and measured robot positions only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROFILES = ('clean', 'static', 'moving', 'occlusion', 'lighting')


def create_environment_model(scene):
    root = ET.fromstring(Path(scene).read_text(encoding='utf-8'))
    world = root.find('worldbody')
    # Live tabletop objects have free joints. The controlled training experiment
    # replaces them with independent noncontact props instead of duplicating them.
    default_objects = [body for body in world.findall('body')
                       if body.get('name', '').startswith('table_object_')]
    for body in default_objects:
        world.remove(body)
    if default_objects:
        for key in root.findall('keyframe/key'):
            positions = key.get('qpos', '').split()
            key.set('qpos', ' '.join(positions[:-7*len(default_objects)]))
    for index in range(3):
        body = ET.SubElement(world, 'body', name=f'env_{index}', mocap='true', pos='10 10 10')
        common = {'group': '2', 'contype': '0', 'conaffinity': '0', 'rgba': '.10 .20 .31 1'}
        if index == 0:
            ET.SubElement(body, 'geom', name='env_0_core', type='capsule',
                          fromto='-.075 0 0 .075 0 0', size='.023', **common)
            ET.SubElement(body, 'geom', name='env_0_stripe', type='box',
                          pos='0 -.024 0', size='.050 .003 .014',
                          **{**common, 'rgba': '.97 .42 .12 1'})
        elif index == 1:
            ET.SubElement(body, 'geom', name='env_1_core', type='box', size='.055 .030 .035', **common)
            ET.SubElement(body, 'geom', name='env_1_stripe', type='box', pos='0 -.032 0',
                          size='.033 .003 .021', **{**common, 'rgba': '.97 .42 .12 1'})
        else:
            ET.SubElement(body, 'geom', name='env_2_core', type='sphere', size='.042',
                          **{**common, 'rgba': '.97 .42 .12 1'})
    xml = ET.tostring(root, encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml), hashlib.sha256(xml.encode()).hexdigest()


class EnvironmentController:
    def __init__(self, model):
        self.model = model
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'env_{i}') for i in range(3)]
        self.mocap_ids = np.array([model.body_mocapid[i] for i in self.body_ids])
        self.geom_ids = [np.flatnonzero(model.geom_bodyid == body) for body in self.body_ids]
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fixed')
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.cam_quat[camera])
        matrix = matrix.reshape(3, 3)
        self.right, self.up, self.forward = matrix[:, 0], matrix[:, 1], -matrix[:, 2]
        self.base_diffuse = model.light_diffuse.copy()
        self.base_ambient = model.light_ambient.copy()
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, 'floor')
        self.base_floor = model.mat_rgba[self.floor_id].copy()
        self.base_colors = model.geom_rgba.copy()
        self.origins = np.full((3, 3), 10.)
        self.velocities = np.zeros((3, 3))
        self.source_time = 0.
        self.profile = 'clean'
        tabletop = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'tabletop')
        self.table_surface = None
        self.support_heights = np.array([.023, .035, .042])
        self.prop_radii = np.array([.11, .07, .05])
        if tabletop >= 0:
            table_body = model.geom_bodyid[tabletop]
            table_center = model.body_pos[table_body]+model.geom_pos[tabletop]
            self.table_surface = float(table_center[2]+model.geom_size[tabletop, 2])
            self.table_center = table_center[:2].copy()
            self.table_half = model.geom_size[tabletop, :2].copy()

    def _table_positions(self, positions):
        if self.table_surface is None or self.profile not in ('static', 'moving', 'occlusion'):
            return positions
        result = positions.copy()
        margin = self.prop_radii[:, None]+.02
        result[:, :2] = np.clip(result[:, :2], self.table_center-self.table_half+margin,
                                self.table_center+self.table_half-margin)
        result[:, 2] = self.table_surface+self.support_heights
        return result

    def configure(self, data, profile, rng):
        if profile not in PROFILES:
            raise ValueError(f'Unknown environment profile: {profile}')
        self.profile = profile
        self.velocities.fill(0)
        self.source_time = float(data.time)
        self.model.light_diffuse[:] = self.base_diffuse
        self.model.light_ambient[:] = self.base_ambient
        self.model.mat_rgba[self.floor_id] = self.base_floor
        self.model.geom_rgba[:] = self.base_colors
        self.origins.fill(10.)
        if profile == 'lighting':
            intensity = float(rng.uniform(.55, 1.35))
            self.model.light_diffuse[:] *= intensity
            self.model.light_ambient[:] *= intensity
            self.model.mat_rgba[self.floor_id, :3] = rng.uniform(.60, 1., 3)
        if profile in ('static', 'moving', 'occlusion'):
            center = np.array([.18, 0., .43])
            for index in range(3):
                if self.table_surface is not None:
                    for attempt in range(100):
                        x = rng.uniform(-.08, .35) if profile == 'occlusion' and index == 0 else rng.uniform(.15, .67)
                        y = rng.uniform(-.36, -.06) if profile == 'occlusion' and index == 0 else rng.uniform(-.37, .37)
                        self.origins[index] = [x, y, self.table_surface+self.support_heights[index]]
                        if all(np.linalg.norm(self.origins[index, :2]-self.origins[prior, :2])
                               > self.prop_radii[index]+self.prop_radii[prior]+.025 for prior in range(index)):
                            break
                    else:
                        raise RuntimeError('Could not place separated objects on the tabletop')
                else:
                    depth = -.22 if profile == 'occlusion' and index == 0 else .18
                    horizontal = rng.uniform(-.15, .18) if depth < 0 else rng.uniform(-.43, .43)
                    vertical = rng.uniform(-.06, .13) if depth < 0 else rng.uniform(-.20, .27)
                    self.origins[index] = center + horizontal*self.right + vertical*self.up + depth*self.forward
                angle = rng.uniform(-np.pi, np.pi)
                rotation_axis = np.array([0., 0., 1.]) if self.table_surface is not None else self.forward
                data.mocap_quat[self.mocap_ids[index]] = np.r_[np.cos(angle/2), rotation_axis*np.sin(angle/2)]
                for geom in self.geom_ids[index]:
                    self.model.geom_rgba[geom, :3] = np.clip(
                        self.base_colors[geom, :3] + rng.uniform(-.035, .035, 3), .02, 1.)
        data.mocap_pos[self.mocap_ids] = self.origins
        mujoco.mj_forward(self.model, data)
        return {'profile': profile, 'distractors_are_kinematic_noncontact': True,
                'moving_independently': profile in ('moving', 'occlusion')}

    def mark_source(self, data):
        self.source_time = float(data.time)

    def begin_probe(self, rng):
        self.velocities.fill(0)
        if self.profile in ('moving', 'occlusion'):
            direction = rng.normal(size=(3, 2))
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
            speed = rng.uniform(.16, .40, (3, 1))
            first, second = ((np.array([1., 0., 0.]), np.array([0., 1., 0.]))
                             if self.table_surface is not None else (self.right, self.up))
            self.velocities = (direction[:, :1]*first + direction[:, 1:]*second)*speed
        return self.velocities.copy()

    def advance(self, data):
        elapsed = float(data.time) - self.source_time
        data.mocap_pos[self.mocap_ids] = self._table_positions(self.origins + elapsed*self.velocities)
        # Props do not contact the arm; updating after physical robot stepping
        # changes the render, without injecting any force into measured joints.
        mujoco.mj_forward(self.model, data)

    def advance_bounded(self, data):
        """Unseen continuous demonstration trajectory; never used for labels."""
        elapsed = float(data.time) - self.source_time
        frequency = np.array([1.2, 1.5, 1.8])[:, None]
        displacement = np.sin(frequency*elapsed) / frequency * self.velocities
        data.mocap_pos[self.mocap_ids] = self._table_positions(self.origins + displacement)
        mujoco.mj_forward(self.model, data)
