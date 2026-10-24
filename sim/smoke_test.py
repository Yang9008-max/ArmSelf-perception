"""Checks the running sandbox's actual dynamics, rendering and saved observations."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


def run(base, require_model=False):
    def get(path):
        with urlopen(base + path, timeout=10) as response:
            return response.read()

    def state():
        return json.loads(get('/api/state'))

    def post(payload):
        request = Request(base + '/api/control', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=25) as response:
            return json.load(response)

    def until(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = state()
            assert not current.get('error'), current
            if predicate(current):
                return current
            time.sleep(.08)
        raise AssertionError('Timed out waiting for expected simulation state')

    try:
        post({'home': True, 'paused': True, 'shadows': True, 'light_angle': -45})
        initial = state()
        assert initial['ready'] and len(initial['joints']) == 7
        t = initial['sim_time']
        time.sleep(.2)
        assert state()['sim_time'] == t, 'Paused physics still advances'
        rgb = np.asarray(Image.open(BytesIO(get('/frame.jpg'))))
        assert rgb.shape == (480, 640, 3) and rgb.std() > 10
        print('PASS: scene loads, camera renders, pause freezes simulation time')

        observation = json.loads(get('/api/observation'))
        camera_bytes = base64.b64decode(observation['camera'].split(',', 1)[1])
        assert camera_bytes == get('/frame.jpg'), 'Standalone camera differs from paired raw frame'
        assert len(observation['measured_q_rad_or_m']) == 7
        assert observation['sim_time'] == t
        assert '摄像头原始画面' in get('/camera').decode('utf-8')
        model_ready = observation['perception']['ready']
        if require_model:
            assert model_ready, observation['perception']
        if model_ready:
            model = observation['perception']
            assert model['frame_id'] == observation['frame_id']
            assert model['sim_time'] == observation['sim_time']
            assert model['inference_ms'] > 0 and len(model['checkpoint_sha256']) == 64
            overlay = np.asarray(Image.open(BytesIO(base64.b64decode(observation['prediction'].split(',', 1)[1]))))
            assert overlay.shape == rgb.shape
            expected_q = np.array([joint['actual'] for joint in initial['joints']])
            expected_q[:6] *= np.pi/180
            expected_q[6] /= 1000
            np.testing.assert_allclose(observation['measured_q_rad_or_m'], expected_q, atol=1e-6)
            print(f"PASS: {model['model']} loaded; synchronized raw camera / prediction / seven measured positions; {model['inference_ms']:.0f} ms inference")

        targets = [j['target'] for j in initial['joints']]
        targets[0] += 15
        targets[6] = 30
        post({'targets': targets, 'paused': False})
        until(lambda s: abs(s['joints'][0]['actual'] - targets[0]) < .3 and
              abs(s['joints'][6]['actual'] - 30) < .2)
        print('PASS: actual arm and gripper positions follow commands')

        post({'auto': True})
        until(lambda s: s['auto'] and s['active_joint'] and abs(s['joints'][0]['actual']) > 3)
        current_targets = [j['target'] for j in state()['joints']]
        post({'targets': current_targets})
        assert not state()['auto']
        print('PASS: automatic exploration moves the arm; manual input stops it')

        invalid = [{'targets': [0] * 6}, {'targets': [float('nan')] * 7},
                   {'targets': [10**400] * 7}, {'targets': [9999] * 7},
                   {'light_angle': 10**400}, {'paused': 'false'}, {'unknown': True}]
        for payload in invalid:
            try:
                post(payload)
            except HTTPError as error:
                assert error.code == 400, error
                assert json.load(error).get('error')
            else:
                raise AssertionError(f'Invalid control accepted: {str(payload)[:60]}')
        print('PASS: malformed, non-finite, oversized and out-of-range controls return HTTP 400')

        post({'home': True, 'paused': True, 'shadows': True})
        shadow_path = Path(post({'snapshot': True})['path'])
        post({'shadows': False})
        clear_path = Path(post({'snapshot': True})['path'])
        shadow = np.asarray(Image.open(shadow_path / 'rgb.png')).astype(np.int16)
        clear = np.asarray(Image.open(clear_path / 'rgb.png')).astype(np.int16)
        mask = np.asarray(Image.open(shadow_path / 'evaluation/body_mask.png')) > 0
        clear_mask = np.asarray(Image.open(clear_path / 'evaluation/body_mask.png')) > 0
        depth = np.load(shadow_path / 'evaluation/depth_m.npy')
        shadow_state = json.loads((shadow_path / 'state.json').read_text(encoding='utf-8'))
        clear_state = json.loads((clear_path / 'state.json').read_text(encoding='utf-8'))
        assert mask.shape == depth.shape == (480, 640)
        assert np.isfinite(depth).all() and (depth > 0).all()
        assert 500 < mask.sum() < mask.size / 2, 'Body mask empty or implausibly large'
        assert np.array_equal(mask, clear_mask), 'Lighting changed geometric body mask'
        assert shadow_state['sim_time'] == clear_state['sim_time']
        assert shadow_state['qpos_rad_or_m'] == clear_state['qpos_rad_or_m']
        assert len(shadow_state['joint_names']) == len(shadow_state['qpos_rad_or_m']) == 8
        assert len(shadow_state['qvel_rad_s_or_m_s']) == 8
        assert all(not name.startswith('table_object_') for name in shadow_state['joint_names'])
        if model_ready:
            probability = np.load(shadow_path / 'prediction/probability.npy')
            predicted_mask = np.asarray(Image.open(shadow_path / 'prediction/body_mask.png')) > 0
            predicted_overlay = np.asarray(Image.open(shadow_path / 'prediction/overlay.png'))
            provenance = json.loads((shadow_path / 'prediction/metadata.json').read_text(encoding='utf-8'))
            assert probability.shape == (144, 192) and np.isfinite(probability).all()
            assert probability.min() >= 0 and probability.max() <= 1
            assert predicted_mask.shape == (480, 640)
            assert provenance['sim_time'] == shadow_state['sim_time']
            np.testing.assert_array_equal(predicted_overlay[~predicted_mask], shadow[~predicted_mask])
            print('PASS: model prediction and provenance saved separately from evaluation truth')
        changed = np.max(np.abs(shadow - clear), axis=2) > 10
        assert (changed & ~mask).sum() > 100, 'No visible cast-shadow change outside body'
        # Coupled fingers should report the same positive displacement.
        assert abs(shadow_state['qpos_rad_or_m'][-1] - shadow_state['qpos_rad_or_m'][-2]) < 1e-4
        print(f'PASS: synchronized RGB/state/depth/body mask; {int((changed & ~mask).sum())} shadow pixels outside body')
        print(f'CHECKED_SNAPSHOT={shadow_path}')
    finally:
        post({'home': True, 'paused': False, 'shadows': True, 'light_angle': -45})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--require-model', action='store_true')
    args = parser.parse_args()
    run(args.url.rstrip('/'), require_model=args.require_model)
