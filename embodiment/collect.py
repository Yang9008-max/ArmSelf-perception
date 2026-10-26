"""Continuous measured-feedback stereo experiment; privileged labels separated."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from perception.collect import HOME
from perception.environment import create_environment_model, EnvironmentController


def stereo_scene(scene, output, width=640, height=480, baseline=.06):
    tree = ET.parse(scene)
    eye = tree.find(".//camera[@name='fixed']")
    if eye is None:
        raise ValueError('Robot eye camera is required')
    # MuJoCo accepts xyaxes/euler/quaternion camera orientation encodings.
    # Its compiled camera quaternion resolves all of them in the parent frame.
    compiled = mujoco.MjModel.from_xml_string(Path(scene).read_text(encoding='utf-8'))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, compiled.cam_quat[compiled.camera('fixed').id])
    right_axis = rotation.reshape(3, 3)[:, 0]
    position = np.fromstring(eye.get('pos'), sep=' ')
    parent = next(node for node in tree.iter() if eye in list(node))
    ET.SubElement(parent, 'camera', **{**eye.attrib, 'name': 'entity_right',
                                     'pos': ' '.join(map(str, position+baseline*right_axis))})
    tree.write(output, encoding='utf-8')
    focal = height/(2*np.tan(np.deg2rad(float(eye.get('fovy')))/2))
    return dict(width=width, height=height, focal_px=float(focal), baseline_m=baseline,
                cx=(width-1)/2, cy=(height-1)/2,
                frame='left camera optical: x right, y down, z forward; metres',
                source='calibrated RGB camera pair only, no robot geometry')


def protocol(seed):
    rng = np.random.default_rng(seed)
    amplitude = np.array([.22, .09, .14, .22, .16, .22, .005])
    records = []
    for _ in range(10):
        records.append(('train', HOME.copy(), True))
    for joint in rng.permutation(7):
        for step in range(32):
            q = HOME.copy()
            q[joint] += amplitude[joint]*np.sin(2*np.pi*step/32)
            records.append(('train', q, True))
    for phase, count in [('train', 100), ('val', 80), ('test', 80)]:
        frequency = rng.uniform(.65, 1.5, 7)
        offset = rng.uniform(-np.pi, np.pi, 7)
        for step in range(count):
            q = HOME+.65*amplitude*np.sin(step*.09*frequency+offset)
            records.append((phase, q, True))
    final = records[-1][1].copy()
    for _ in range(24):
        records.append(('hold', final.copy(), True))
    for step in range(48):
        q = final+.35*amplitude*np.sin(step*.10*np.arange(1, 8)**.3)
        records.append(('blind', q, False))
    return records


def evaluation_protocol(seed):
    """New local motions with a home anchor; contains no fitting/selection frames."""
    return [('test', HOME.copy(), True) for _ in range(10)] + [
        row for row in protocol(seed) if row[0] not in ('train', 'val')]


def collect(output, scene='sim/scenes/robot-eye-v6.xml', seed=20261210, evaluation_only=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh experiment data directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('observations', 'evaluation'):
        (output/name).mkdir()
    calibration = stereo_scene(scene, output/'stereo.xml')
    model, environment_hash = create_environment_model(output/'stereo.xml')
    data = mujoco.MjData(model)
    q_indices = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    environment = EnvironmentController(model)
    rng = np.random.default_rng(seed+1)
    environment.configure(data, 'moving', rng)
    mujoco.mj_step(model, data, nstep=round(.8/model.opt.timestep))
    environment.mark_source(data)
    environment.begin_probe(rng)
    rows = []
    renderer = mujoco.Renderer(model, height=calibration['height'], width=calibration['width'])
    try:
        schedule = evaluation_protocol(seed) if evaluation_only else protocol(seed)
        for index, (phase, target, available) in enumerate(schedule):
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=round(.1/model.opt.timestep))
            environment.advance_bounded(data)
            mujoco.mj_forward(model, data)
            q = data.qpos[q_indices].astype(np.float32).copy()
            images = []
            for camera in ('fixed', 'entity_right'):
                renderer.update_scene(data, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                images.append(renderer.render().copy())
            name = f'{index:05d}.npz'
            np.savez_compressed(output/'observations'/name,
                                left=images[0] if available else np.zeros_like(images[0]),
                                right=images[1] if available else np.zeros_like(images[1]),
                                q=q, time=np.float64(data.time), camera_available=np.bool_(available))
            # Identity/depth never appear in observations, even for training frames.
            renderer.update_scene(data, camera='fixed')
            renderer.enable_segmentation_rendering()
            labels = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            is_geom = labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            body = is_geom & np.isin(labels[..., 0], np.flatnonzero(model.geom_group == 1))
            props = is_geom & np.isin(labels[..., 0], np.flatnonzero(model.geom_group == 2))
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            evaluation = dict(body_mask=body, prop_mask=props, depth=depth)
            if not available:
                evaluation.update(oracle_left=images[0], oracle_right=images[1])
            np.savez_compressed(output/'evaluation'/name, **evaluation)
            rows.append(dict(file=name, phase=phase, camera_available=available, time=float(data.time)))
            if index % 50 == 0:
                print(f'Collected {index+1} stereo frames: {phase}', flush=True)
    finally:
        renderer.close()
    metadata = dict(format='body-entity-stereo-v1', seed=seed, calibration=calibration,
                    evaluation_only=evaluation_only,
                    scene_sha256=hashlib.sha256(Path(scene).read_bytes()).hexdigest(),
                    environment_xml_sha256=environment_hash, frames=rows,
                    feedback='measured q after physical stepping, no commands in learner inputs',
                    missing_camera='RGB zeroed in observations; hidden raw images only under evaluation',
                    limitations=['fixed calibrated stereo pair', 'ideal joint feedback', 'sparse local motion range'])
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene', default='sim/scenes/robot-eye-v6.xml')
    parser.add_argument('--seed', type=int, default=20261210)
    parser.add_argument('--evaluation-only', action='store_true')
    collect(**vars(parser.parse_args()))
