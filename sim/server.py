"""Local MuJoCo perception playground. All physics and GL work has one owner."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, TimeoutError as FutureTimeout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from urllib.parse import urlsplit

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "outputs/perception/v6-eye/model/model.pt"
WIDTH, HEIGHT = 640, 480
LABELS = ["J1 底部旋转", "J2 肩关节", "J3 肘关节", "J4 腕部旋转", "J5 腕部俯仰", "J6 末端旋转", "夹爪单侧行程"]


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_control(payload: object) -> dict:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("需要非空 JSON 对象")
    allowed = {"targets", "paused", "auto", "shadows", "light_angle", "home", "snapshot"}
    if set(payload) - allowed:
        raise ValueError("未知控制字段")
    for key in ("paused", "auto", "shadows", "home", "snapshot"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"{key} 必须是布尔值")
    if "targets" in payload:
        values = payload["targets"]
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError("targets 必须包含六个关节角度和一个夹爪行程")
        if any(not finite_number(v) for v in values):
            raise ValueError("关节目标必须是有限数值")
    if "light_angle" in payload:
        angle = payload["light_angle"]
        if not finite_number(angle) or not -180 <= angle <= 180:
            raise ValueError("光照角度必须在 -180 到 180 度之间")
    return payload


class Simulation:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.commands: queue.Queue = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.state = {"ready": False, "error": None}
        self.jpeg = b""
        self.observation = {}
        self.frame_id = 0
        self.checkpoint = Path(checkpoint)
        self.predictor = None
        self.perception_status = {"ready": False, "model": "模型", "error": None}
        self.thread = threading.Thread(target=self._run, name="mujoco-render", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(30):
            self.stop.set()
            raise RuntimeError("MuJoCo 初始化超时")
        if self.state.get("error"):
            raise RuntimeError(self.state["error"])

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)

    def read_state(self):
        with self.lock:
            return dict(self.state)

    def read_frame(self):
        with self.lock:
            return self.jpeg

    def read_observation(self):
        with self.lock:
            return dict(self.observation)

    def command(self, payload):
        payload = validate_control(payload)
        if self.stop.is_set() or self.read_state().get("error"):
            raise RuntimeError("仿真未运行")
        future = Future()
        try:
            self.commands.put_nowait((payload, future))
        except queue.Full:
            raise RuntimeError("控制请求过多，请稍后重试") from None
        try:
            return future.result(timeout=20)
        except FutureTimeout:
            future.cancel()
            raise RuntimeError("仿真响应超时，请检查服务日志") from None

    def _load(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_id < 0 or self.model.nu != 7:
            raise RuntimeError("场景缺少 home 姿态或七个执行器")
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.camera_id = self.model.camera("fixed").id
        self.light_id = self.model.light("key").id
        self.scales = np.array([180 / np.pi] * 6 + [1000.0])
        self.limits = self.model.actuator_ctrlrange * self.scales[:, None]
        self.joint_ids = self.model.actuator_trnid[:, 0]
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        # Export only robot feedback, including the passive coupled finger.
        # Free-object poses and quaternion coordinates are environment state.
        self.feedback_joint_ids = np.unique(np.r_[self.joint_ids, self.model.joint("finger_right").id])
        self.feedback_qpos_ids = self.model.jnt_qposadr[self.feedback_joint_ids]
        self.feedback_qvel_ids = self.model.jnt_dofadr[self.feedback_joint_ids]
        self.body_geom_ids = np.flatnonzero(self.model.geom_group == 1)
        self.paused = False
        self.auto = False
        self.shadows = True
        self.light_angle = -45.0
        self.fps = 0.0
        self.active_joint = None
        self.last_snapshot = None
        self._home()
        self._light()
        try:
            # Also support launching as `python sim/server.py` from the repo.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from perception.live import LiveBodyPredictor
            self.predictor = LiveBodyPredictor(self.checkpoint)
            self.perception_status = {**self.predictor.metadata, "ready": True,
                                      "error": None}
        except Exception as exc:
            self.perception_status = {"ready": False, "model": "模型", "error": str(exc)}
            print(f"Perception unavailable; camera and controls remain active: {exc}", flush=True)

    def _predict(self, rgb, measured_q):
        if self.predictor is None:
            return None
        try:
            return self.predictor.predict(rgb, measured_q)
        except Exception as exc:
            self.perception_status = {**self.perception_status, "ready": False, "error": str(exc)}
            self.predictor = None
            print(f"Perception stopped; camera and controls remain active: {exc}", flush=True)
            return None

    def _home(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        mujoco.mj_forward(self.model, self.data)
        self.targets = self.data.ctrl.copy()
        self.auto = False
        self.active_joint = None

    def _light(self):
        angle = math.radians(self.light_angle)
        position = np.array([1.7 * math.cos(angle), 1.7 * math.sin(angle), 2.3])
        self.model.light_pos[self.light_id] = position
        direction = self.model.stat.center - position
        self.model.light_dir[self.light_id] = direction / np.linalg.norm(direction)
        self.model.light_castshadow[self.light_id] = self.shadows

    def _apply(self, payload):
        # Check the complete command before mutating any state.
        if "targets" in payload:
            values = np.asarray(payload["targets"], dtype=float)
            if np.any(values < self.limits[:, 0] - 1e-6) or np.any(values > self.limits[:, 1] + 1e-6):
                raise ValueError("关节目标超出允许范围")
        if payload.get("home"):
            self._home()
        if "auto" in payload:
            if payload["auto"] and not self.auto:
                self._home()
                self.auto_origin = self.targets.copy()
                self.auto_start = self.data.time
                self.paused = False
            self.auto = payload["auto"]
            if not self.auto:
                self.active_joint = None
        if "targets" in payload:
            self.targets = np.asarray(payload["targets"]) / self.scales
            self.auto = False
            self.active_joint = None
        for key in ("paused", "shadows", "light_angle"):
            if key in payload:
                setattr(self, key, payload[key])
        self._light()
        self.data.ctrl[:] = self.targets
        if payload.get("snapshot"):
            return {"ok": True, "path": self._snapshot()}
        return {"ok": True}

    def _step(self):
        if self.auto:
            elapsed = self.data.time - self.auto_start
            index = int(elapsed / 4.5) % 7
            phase = elapsed % 4.5
            # One full sinusoidal excursion, followed by a 0.5 second rest.
            wave = math.sin(2 * math.pi * phase / 4) if phase < 4 else 0.0
            amplitudes = np.deg2rad([18, 12, 16, 20, 16, 25]).tolist() + [0.009]
            self.targets[:] = self.auto_origin
            room = min(self.auto_origin[index] - self.model.actuator_ctrlrange[index, 0],
                       self.model.actuator_ctrlrange[index, 1] - self.auto_origin[index])
            self.targets[index] += min(amplitudes[index], max(0, room)) * wave
            self.active_joint = LABELS[index]
        self.data.ctrl[:] = self.targets
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
            raise RuntimeError("仿真出现数值异常；请重启并检查姿态或模型参数")

    def _render_rgb(self):
        # mj_step leaves derived positions at the preceding step; refresh before
        # rendering so images and qpos in the exported observation agree.
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="fixed")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        return self.renderer.render().copy()

    def _state(self):
        targets = self.targets * self.scales
        actual = self.data.qpos[self.qpos_ids] * self.scales
        joints = [dict(name=self.model.actuator(i).name, label=LABELS[i],
                       min=float(self.limits[i, 0]), max=float(self.limits[i, 1]),
                       target=float(targets[i]), actual=float(actual[i]),
                       unit="deg" if i < 6 else "mm") for i in range(7)]
        return dict(ready=True, sim_time=float(self.data.time), paused=self.paused,
                    auto=self.auto, shadows=self.shadows, light_angle=self.light_angle,
                    fps=round(self.fps, 1), joints=joints, active_joint=self.active_joint,
                    last_snapshot=self.last_snapshot, perception=self.perception_status, error=None)

    @staticmethod
    def _encode_jpeg(rgb):
        image = BytesIO()
        Image.fromarray(rgb).save(image, format="JPEG", quality=88)
        return image.getvalue()

    def _publish(self):
        rgb = self._render_rgb()
        measured_q = self.data.qpos[self.qpos_ids].copy()
        prediction = self._predict(rgb, measured_q)
        camera_jpeg = self._encode_jpeg(rgb)
        self.frame_id += 1
        sim_time = float(self.data.time)
        if prediction is not None:
            self.perception_status = {**self.perception_status, "frame_id": self.frame_id,
                                      "sim_time": sim_time,
                                      "inference_ms": round(prediction["inference_ms"], 2),
                                      "predicted_body_pixels": int(prediction["mask"].sum())}
        observation = {"frame_id": self.frame_id, "sim_time": sim_time,
                       "measured_q_rad_or_m": measured_q.tolist(),
                       "camera": "data:image/jpeg;base64,"+base64.b64encode(camera_jpeg).decode("ascii"),
                       "prediction": None, "perception": self.perception_status}
        if prediction is not None:
            observation["prediction"] = "data:image/jpeg;base64,"+base64.b64encode(
                self._encode_jpeg(prediction["overlay"])).decode("ascii")
        with self.lock:
            self.jpeg = camera_jpeg
            self.observation = observation
            self.state = self._state()

    def _snapshot(self):
        rgb = self._render_rgb()
        measured_q = self.data.qpos[self.qpos_ids].copy()
        prediction = self._predict(rgb, measured_q)
        try:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        try:
            self.renderer.enable_segmentation_rendering()
            segmentation = self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()
        mask = ((segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                np.isin(segmentation[:, :, 0], self.body_geom_ids))
        output = ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        evaluation = output / "evaluation"
        evaluation.mkdir(parents=True)
        Image.fromarray(rgb).save(output / "rgb.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(evaluation / "body_mask.png")
        np.save(evaluation / "depth_m.npy", depth)
        observation = self._state()
        if prediction is not None:
            prediction_dir = output / "prediction"
            prediction_dir.mkdir()
            Image.fromarray(prediction["overlay"]).save(prediction_dir / "overlay.png")
            Image.fromarray(prediction["mask"].astype(np.uint8)*255).save(prediction_dir / "body_mask.png")
            np.save(prediction_dir / "probability.npy", prediction["probability"])
            metadata = {**self.perception_status, "sim_time": float(self.data.time),
                        "source": "snapshot", "inference_ms": prediction["inference_ms"],
                        "measured_q_rad_or_m": measured_q.tolist(),
                        "predicted_body_pixels": int(prediction["mask"].sum())}
            metadata.pop("frame_id", None)
            (prediction_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            observation["perception"] = metadata
        observation.update(schema_version=1, mujoco_version=mujoco.__version__,
                           qpos_rad_or_m=self.data.qpos[self.feedback_qpos_ids].tolist(),
                           qvel_rad_s_or_m_s=self.data.qvel[self.feedback_qvel_ids].tolist(),
                           command_rad_or_m=self.data.ctrl.tolist(),
                           joint_names=[self.model.joint(int(i)).name for i in self.feedback_joint_ids],
                           image_width=WIDTH, image_height=HEIGHT,
                           camera={"name": "fixed", "vertical_fov_deg": float(self.model.cam_fovy[self.camera_id]),
                                   "world_position_m": self.data.cam_xpos[self.camera_id].tolist(),
                                   "world_rotation_matrix": self.data.cam_xmat[self.camera_id].reshape(3, 3).tolist()},
                           evaluation_only={"body_mask.png": "Visible moving arm/gripper geom surfaces; fixed base, background and cast shadows excluded.",
                                            "depth_m.npy": "Perfect simulator camera-axis depth in metres, NOT estimated stereo depth. Background may equal the far clipping plane."})
        (output / "state.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_snapshot = str(output)
        return self.last_snapshot

    def _run(self):
        pending = None
        try:
            self._load()
            self._publish()
            self.ready.set()
            last_tick = last_render = time.monotonic()
            budget = 0.0
            while not self.stop.is_set():
                changed = False
                for _ in range(16):
                    try:
                        payload, pending = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if not pending.set_running_or_notify_cancel():
                        continue
                    try:
                        result = self._apply(payload)
                        self._publish()
                        pending.set_result(result)
                        changed = True
                    except ValueError as exc:
                        pending.set_exception(exc)
                    pending = None
                now = time.monotonic()
                elapsed = min(now - last_tick, 0.1)
                last_tick = now
                if self.paused:
                    budget = 0.0
                else:
                    budget += elapsed
                    while budget >= self.model.opt.timestep:
                        self._step()
                        budget -= self.model.opt.timestep
                if now - last_render >= 0.05:
                    self.fps = 1 / max(now - last_render, 1e-6)
                    self._publish()
                    last_render = now
                if not changed:
                    self.stop.wait(0.002)
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.state = {**self.state, "ready": False, "error": str(exc)}
            if pending is not None and not pending.done():
                pending.set_exception(RuntimeError(str(exc)))
        finally:
            self.stop.set()
            self.ready.set()
            while not self.commands.empty():
                _, future = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError("仿真已经停止"))
            if hasattr(self, "renderer"):
                self.renderer.close()


def make_handler(simulation):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            if args and str(args[1] if len(args) > 1 else "") not in ("200", "304"):
                super().log_message(format, *args)

        def reply(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def json_reply(self, status, data):
            self.reply(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.reply(200, (ROOT / "sim" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self.json_reply(200, simulation.read_state())
            elif path == "/api/observation":
                observation = simulation.read_observation()
                self.json_reply(200 if observation else 503, observation)
            elif path == "/camera":
                self.reply(200, (ROOT / "sim" / "camera.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/frame.jpg":
                frame = simulation.read_frame()
                self.reply(200 if frame else 503, frame, "image/jpeg")
            elif path == "/favicon.ico":
                self.reply(204, b"", "image/x-icon")
            else:
                self.json_reply(404, {"error": "未找到页面"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/control":
                self.json_reply(404, {"error": "未找到接口"})
                return
            # Only same-origin JSON requests can mutate this local service.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json_reply(403, {"error": "拒绝跨站控制请求"})
                return
            if self.headers.get_content_type() != "application/json":
                self.json_reply(415, {"error": "需要 application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("请求长度无效")
                payload = json.loads(self.rfile.read(length))
                self.json_reply(200, simulation.command(payload))
            except (ValueError, UnicodeError) as exc:
                self.json_reply(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.json_reply(503, {"error": str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="MuJoCo local perception sandbox")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    simulation = Simulation(checkpoint=args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(simulation))
    server.daemon_threads = True
    try:
        simulation.start()
        address = f"http://127.0.0.1:{args.port}"
        print(f"MuJoCo {mujoco.__version__} ready: {address}", flush=True)
        print("Keep this window running. Ctrl+C stops the simulation.", flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(address)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        simulation.close()


if __name__ == "__main__":
    main()
