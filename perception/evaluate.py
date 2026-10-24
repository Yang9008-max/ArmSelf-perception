"""Evaluate a selected checkpoint; privileged body masks are read only here."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Binary segmentation metrics with explicit empty-mask conventions."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Mask shapes differ: {prediction.shape} and {target.shape}")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    return _scores(tp, fp, fn, tn)


def _scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float(fn == 0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _aggregate_masks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "macro": {}, "micro": {}}
    macro = {key: float(np.mean([row[key] for row in rows])) for key in ("iou", "precision", "recall")}
    return {
        "count": len(rows), "macro": macro,
        "micro": _scores(*(sum(row[k] for row in rows) for k in ("tp", "fp", "fn", "tn"))),
        "iou_min": min(row["iou"] for row in rows),
        "iou_median": float(np.median([row["iou"] for row in rows])),
        "iou_max": max(row["iou"] for row in rows),
    }


def distractor_metrics(prediction, distractor):
    distractor = np.asarray(distractor, dtype=bool)
    pixels = int(distractor.sum())
    mistaken = int((np.asarray(prediction, dtype=bool) & distractor).sum())
    return {'pixels': pixels, 'predicted_body_pixels': mistaken,
            'false_positive_rate': mistaken/pixels if pixels else None}


def _aggregate_distractors(rows):
    pixels = sum(row['pixels'] for row in rows)
    mistaken = sum(row['predicted_body_pixels'] for row in rows)
    return {'pixels': pixels, 'predicted_body_pixels': mistaken,
            'false_positive_rate': mistaken/pixels if pixels else None}


def _error_summary(error: np.ndarray, selector: np.ndarray) -> dict[str, Any]:
    selected = error[selector]
    return {"pixels": int(selected.size), "sum_error": float(selected.sum(dtype=np.float64)),
            "epe": float(selected.mean()) if selected.size else None}


def _motion_scores(jacobian: np.ndarray, cached: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, Any]:
    flow = np.asarray(cached["flow"], dtype=np.float32)
    dq = np.asarray(cached["dq_scaled"], dtype=np.float32)
    valid = np.asarray(cached["valid"], dtype=bool)
    if flow.shape[0] < 2 or flow.shape[-1] != 2:
        raise ValueError("Cache flow must have shape (signed probes, H, W, 2)")
    negative = np.arange(1, len(flow), 2)
    flow, dq, valid = flow[negative], dq[negative], valid[negative]
    valid &= np.isfinite(flow).all(axis=-1)
    magnitude = np.linalg.norm(flow, axis=-1)
    predicted = np.einsum("kchw,nk->nhwc", jacobian, dq, optimize=True)
    permutation = np.arange(len(dq))
    if len(dq) > 1:
        # Avoid a fixed-point pairing accidentally rewarding the shuffled baseline.
        for _ in range(100):
            permutation = rng.permutation(len(dq))
            if np.all(permutation != np.arange(len(dq))):
                break
        else:
            permutation = np.roll(np.arange(len(dq)), 1)
    shuffled = np.einsum("kchw,nk->nhwc", jacobian, dq[permutation], optimize=True)
    result: dict[str, Any] = {"probe_indices": negative.tolist(), "shuffle_permutation": permutation.tolist()}
    for name, estimate in (("model", predicted), ("zero_motion", np.zeros_like(flow)), ("shuffled_dq", shuffled)):
        error = np.linalg.norm(estimate - flow, axis=-1)
        result[name] = {
            "all": _error_summary(error, valid),
            "moving": _error_summary(error, valid & (magnitude > 0.35)),
            "static": _error_summary(error, valid & (magnitude <= 0.35)),
        }
    return result


def _aggregate_motion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for method in ("model", "zero_motion", "shuffled_dq"):
        result[method] = {}
        for region in ("all", "moving", "static"):
            count = sum(row[method][region]["pixels"] for row in rows)
            error_sum = sum(row[method][region]["sum_error"] for row in rows)
            result[method][region] = {"pixels": count, "epe": error_sum / count if count else None}
    return result


def _label_scores(labels: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if labels.shape != truth.shape or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Partial labels must match the mask and contain only -1, 0, 1")
    positive, negative = labels == 1, labels == 0
    return {"positive_pixels": int(positive.sum()), "negative_pixels": int(negative.sum()),
            "total_pixels": int(labels.size), "positive_body_pixels": int((positive & truth).sum()),
            "negative_body_pixels": int((negative & truth).sum())}


def _finish_label_scores(counts: dict[str, int]) -> dict[str, Any]:
    p, n, total = counts["positive_pixels"], counts["negative_pixels"], counts["total_pixels"]
    return {**counts, "positive_precision": counts["positive_body_pixels"] / p if p else None,
            "negative_body_contamination": counts["negative_body_pixels"] / n if n else None,
            "positive_coverage": p / total if total else None,
            "negative_coverage": n / total if total else None,
            "known_coverage": (p + n) / total if total else None}


def _font(size: int) -> ImageFont.ImageFont:
    for font_path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _bound_preprocessing(saved: dict[str, Any], metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Bind inference normalization and motion units to the selected weights."""
    preprocessing = saved.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Checkpoint lacks bound preprocessing; use the completed final evaluation checkpoint, not an older motion-only checkpoint")
    channels = saved.get("model_config", {}).get("channels", 7)
    values = {}
    for key in ("q_center", "q_scale", "probe_scales"):
        if key not in preprocessing or key not in metadata:
            raise ValueError(f"Checkpoint and dataset must both define preprocessing field {key!r}")
        checkpoint_value = np.asarray(preprocessing[key], dtype=np.float64)
        dataset_value = np.asarray(metadata[key], dtype=np.float64)
        if checkpoint_value.shape != (channels,) or dataset_value.shape != (channels,):
            raise ValueError(f"Preprocessing field {key!r} must contain {channels} values")
        if not np.isfinite(checkpoint_value).all() or not np.isfinite(dataset_value).all():
            raise ValueError(f"Preprocessing field {key!r} contains non-finite values")
        if key != "q_center" and (np.any(checkpoint_value <= 0) or np.any(dataset_value <= 0)):
            raise ValueError(f"Preprocessing field {key!r} must be positive")
        if not np.allclose(checkpoint_value, dataset_value, rtol=1e-7, atol=1e-9):
            raise ValueError(f"Dataset {key!r} differs from checkpoint preprocessing; normalization and motion units must match the selected weights")
        values[key] = checkpoint_value.astype(np.float32)
    return values["q_center"], values["q_scale"]


def _examples_image(examples: list[dict[str, Any]], output: Path) -> list[str]:
    ranked = sorted(examples, key=lambda item: (item["source_metrics"]["iou"], item["id"]))
    selected = [ranked[i] for i in sorted(set((0, len(ranked) // 2, len(ranked) - 1)))]
    width, height = selected[0]["rgb"].shape[1], selected[0]["rgb"].shape[0]
    scale = 2
    row_height = height * scale + 42
    panel = Image.new("RGB", (3 * width * scale, len(selected) * 2 * row_height + 36), "#17202c")
    draw = ImageDraw.Draw(panel)
    font = _font(17)
    for col, caption in enumerate(("Camera RGB", "Predicted body (threshold 0.5)", "Evaluation truth (never trained)")):
        draw.text((col * width * scale + 8, 8), caption, font=font, fill="white")
    row = 0
    for sample in selected:
        for name, rgb_key, pred_key, truth_key, metric_key in (
            ("source", "rgb", "prediction", "truth", "source_metrics"),
            ("stopped after motion", "stop_rgb", "stop_prediction", "stop_truth", "stop_metrics"),
        ):
            top = 36 + row * row_height
            caption = f'{sample["id"]} | {name} | IoU {sample[metric_key]["iou"]:.3f}'
            draw.text((8, top + 5), caption, font=font, fill="white")
            panels = [Image.fromarray(sample[rgb_key]),
                      Image.fromarray(sample[pred_key].astype(np.uint8) * 255).convert("RGB"),
                      Image.fromarray(sample[truth_key].astype(np.uint8) * 255).convert("RGB")]
            for col, picture in enumerate(panels):
                panel.paste(picture.resize((width * scale, height * scale), Image.Resampling.NEAREST),
                            (col * width * scale, top + 38))
            row += 1
    panel.save(output)
    return [item["id"] for item in selected]


def _number(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return "无有效像素"
    return f"{value * 100:.1f}%" if percentage else f"{value:.4f}"


def _html_report(metrics: dict[str, Any], image_path: Path) -> str:
    source = metrics["source"]["macro"]
    stopped = metrics["stopped_after_motion"]["macro"]
    segmentation_rows = "".join(
        f"<tr><td>{name}</td><td>{_number(row['iou'], True)}</td><td>{_number(row['precision'], True)}</td><td>{_number(row['recall'], True)}</td></tr>"
        for name, row in (("初始姿态", source), ("运动后保持的姿态", stopped))
    )
    motion_html = "<p>本次未提供图像监督缓存，未计算运动预测和伪标签指标。</p>"
    if metrics.get("motion"):
        motion_rows = "".join(
            f"<tr><td>{label}</td><td>{_number(metrics['motion'][key]['moving']['epe'])}</td><td>{_number(metrics['motion'][key]['static']['epe'])}</td></tr>"
            for key, label in (("model", "模型 + 正确反馈变化"), ("zero_motion", "全部预测不动"), ("shuffled_dq", "同姿态探测的反馈变化打乱"))
        )
        motion_html = f"<table><thead><tr><th>方法</th><th>运动区域 EPE（像素）</th><th>静止区域 EPE（像素）</th></tr></thead><tbody>{motion_rows}</tbody></table><p>仅评估负向探测；运动区域由有效光流幅度 &gt; 0.35 像素定义。目标来自 RGB 光流估计，不是真实物理光流。打乱的是 Δq 与视觉变化的配对，不能单独证明网络充分使用了当前姿态反馈。</p>"
    label_html = ""
    if metrics.get("partial_labels"):
        labels = metrics["partial_labels"]
        label_html = f"<h2>自动候选区域：仅在评估时与真值比较</h2><p>正标签精确率 {_number(labels['positive_precision'], True)}；负标签中的身体比例 {_number(labels['negative_body_contamination'], True)}；已知标签覆盖率 {_number(labels['known_coverage'], True)}。这些真值比较结果不参与伪标签生成或训练。</p>"
        if metrics.get('training', {}).get('boundary_refinement', {}).get('enabled'):
            label_html += '<p>本次将动作/静态背景种子交给逐图像 GrabCut，利用图像外观推断原先未知的像素。这些补全结果是弱标签；覆盖率达到 100% 也不代表它们全部确定或正确。原始种子另存于缓存 seed_labels。</p>'
    architecture_html = ''
    if metrics.get('model_config', {}).get('separate_motion'):
        architecture_html = '<p>此检查点包含两套独立权重：身体分支负责分割，冻结的运动分支负责像素响应 J。分割训练不会改写运动分支；这是一套组合模型，来源哈希见完整记录。</p>'
    if metrics.get('model_config', {}).get('body_support'):
        architecture_html = '<p>此检查点包含三套网络权重：新身体分支、冻结的 V4 身体候选分支、冻结的运动响应分支。两个身体 logit 取较小值；在相同阈值下等价于掩膜取交集。它能阻止新分支向旧候选范围外扩张，也无法补回旧模型漏掉的身体像素。来源哈希见完整记录。</p>'
    environment_html = ''
    if metrics.get('by_environment'):
        environment_rows = ''
        for profile, group in metrics['by_environment'].items():
            rate = group['distractors']['false_positive_rate']
            environment_rows += (f"<tr><td>{html.escape(profile)}</td><td>{group['source']['count']}</td>"
                                 f"<td>{_number(group['source']['macro']['iou'], True)}</td>"
                                 f"<td>{_number(group['stopped_after_motion']['macro']['iou'], True)}</td>"
                                 f"<td>{_number(rate, True)}</td></tr>")
        environment_html = f'<h2>环境干扰分组</h2><table><tr><th>场景</th><th>姿态组</th><th>身体 IoU</th><th>保持姿态 IoU</th><th>杂物像素被当成身体</th></tr>{environment_rows}</table><p>杂物率只统计可见杂物像素，干净/纯光照场景无杂物时无此指标。遮挡场景只评价当前可见身体。杂物几何分组只用于评价。</p>'
        if metrics.get('null_controls'):
            held = metrics['null_controls']
            environment_html += (f"<p>单独的机械臂保持、环境继续变化对照，共 {held['body']['count']} 帧：身体 IoU "
                                 f"{_number(held['body']['macro']['iou'], True)}，杂物误判率 "
                                 f"{_number(held['distractors']['false_positive_rate'], True)}。</p>")
    feedback_html = ""
    if metrics.get("current_q_perturbation"):
        diagnostic = metrics["current_q_perturbation"]["summary"]
        feedback_rows = "".join(
            f"<tr><td>{label}</td><td>{_number(diagnostic[key]['segmentation']['macro']['iou'], True)}</td><td>{_number(diagnostic[key]['moving_flow']['epe'])}</td></tr>"
            for key, label in (("correct_q", "正确当前 q"), ("shuffled_q", "跨姿态打乱 q"), ("center_q", "固定为归一化中心"))
        )
        changed = diagnostic["shuffled_q"]["mask_change_fraction_of_prediction_union"]
        feedback_html = f"<h2>当前反馈是否影响分割</h2><table><thead><tr><th>当前位置输入</th><th>身体 IoU</th><th>运动区域 EPE</th></tr></thead><tbody>{feedback_rows}</tbody></table><p>固定图像与正确 Δq，仅改变送入网络的当前 q。打乱 q 后，变化像素占预测身体并集 {_number(changed, True)}。这一诊断衡量冻结模型的输入依赖，不是重新训练的图像单模态对照；若分割几乎不变，就不能宣称当前反馈融合对分割有效。</p>"
    details = html.escape(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    shadow_note = "开启投影阴影" if metrics.get("shadows", False) else "关闭投影阴影"
    appearance_note = ('相机固定，场景加入有限种类的静态/独立运动杂物、前景遮挡配置，以及组间照明与地面外观变化。杂物为无接触的可视干扰。'
                       if metrics.get('by_environment') else '相机和场景外观固定。')
    remaining_note = ('本轮光照变化只覆盖指定范围、且组内探测照明不变；未知工具、相机移动、动态照明与影子区分仍未验证。'
                      if metrics.get('by_environment') else '未知工具、相机移动、光照变化或影子区分尚未验证。')
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机械臂自感知：实测结果</title>
<style>body{{margin:0;background:#111827;color:#e5e7eb;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:1180px;margin:auto;padding:32px 24px}}h1,h2{{line-height:1.3}}h2{{margin-top:32px;color:#93c5fd}}p{{max-width:1000px}}table{{border-collapse:collapse;width:100%;background:#1f2937}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #374151}}img{{width:100%;height:auto;border:1px solid #374151}}code,pre{{font-family:Consolas,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#0b1220;padding:16px}}.note{{padding:14px 18px;background:#243147;border-left:4px solid #60a5fa}}small{{color:#a8b3c5}}</style></head>
<body><main><h1>机械臂自感知：实测结果</h1>
<p>固定相机图像与实际仿真关节反馈输入模型，输出可见身体区域。数据分组：<strong>{html.escape(metrics['split'])}</strong>，共 <strong>{metrics['source']['count']}</strong> 个独立初始姿态。阈值预先固定为 0.5。</p>
<p class="note">这是通用六旋转关节加夹爪的 7 通道 MuJoCo 原型，当前{shadow_note}。没有完成 SO-ARM 或真实硬件验证。训练未读取 CAD、几何 ID、深度或身体真值掩膜；仿真掩膜只由本评估器读取。</p>
{architecture_html}
<h2>身体分割</h2><p>下表为逐姿态平均；像素汇总指标与每个姿态的结果见末尾完整记录。</p><table><thead><tr><th>帧类型</th><th>IoU</th><th>精确率</th><th>召回率</th></tr></thead><tbody>{segmentation_rows}</tbody></table>
<p>停止帧来自正向肩关节运动后继续保持目标的位置，重新执行物理仿真并采集图像和反馈，并非将初始帧改名。停止与初始帧姿态不同，因此两者指标差异同时受到姿态变化影响。</p>
<h2>视觉运动预测</h2>{motion_html}{feedback_html}{label_html}
{environment_html}
<h2>失败、中间与较好样本</h2><p>按初始姿态 IoU 排序，确定性选取最低、中位和最高样本；每个样本同时展示初始帧及运动后停止帧。白色代表身体区域。</p><img alt="代表性图像、模型预测和评估真值，含停止帧" src="data:image/png;base64,{encoded}">
<h2>本实验的边界</h2><p>{appearance_note}关节反馈是无噪声、无延迟的理想仿真测量。部分负标签使用仅从训练 RGB 拟合并冻结的背景参考，假设姿态探索使背景反复可见；持续可见而缺少外观变化的身体表面仍可能被误判为背景。光流会在边界、弱纹理与遮挡处出错。身体面积较小，平均背景误差可能掩盖身体上的错误；环境独立运动还会污染原始光流指标，因此应优先查看身体和杂物的分组指标。每次推理仅用当前图像和位置，没有学习时间记忆。停止姿态评估尚不能等同于真实世界的长期身体记忆。{remaining_note}</p>
<details><summary>完整指标、训练记录与可复现信息</summary><pre>{details}</pre></details><small>此文件内嵌全部示例图，可单独打开，无外部网络依赖。</small></main></body></html>"""


def evaluate_checkpoint(data: str | Path, checkpoint: str | Path, output: str | Path,
                        split: str = "test", cache: str | Path | None = None, threads: int = 4,
                        feedback_diagnostic: str | Path | None = None) -> dict[str, Any]:
    """Run selected weights against held-out observations, then score with masks."""
    from .model import BodyPerception

    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    data, checkpoint, output = Path(data), Path(checkpoint), Path(output)
    cache = Path(cache) if cache is not None else None
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected evaluation checkpoint does not exist: {checkpoint}; complete training and select its final checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    q_center, q_scale = _bound_preprocessing(saved, metadata)
    model = BodyPerception(**saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    files = sorted((data / "observations" / split).glob("*.npz"))
    if not files:
        raise ValueError(f"No observations for split {split!r}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261024)
    rows, examples, motions, label_rows = [], [], [], []
    null_body_scores, null_distractor_scores = [], []
    profiles = {item['id']: item.get('environment_profile', 'clean') for item in metadata.get('samples', [])}

    def infer(rgb: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None] / 255.0
        q_tensor = torch.from_numpy(np.asarray((q - q_center) / q_scale, dtype=np.float32))[None]
        prediction = model(image_tensor, q_tensor)
        mask = (torch.sigmoid(prediction["logits"])[0, 0] >= 0.5).cpu().numpy()
        jacobian = prediction["jacobian"][0].cpu().numpy()
        if not np.isfinite(jacobian).all() or not torch.isfinite(prediction["logits"]).all():
            raise ValueError("Model produced non-finite outputs")
        return mask, jacobian

    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as observation:
                rgb, q = observation["rgb"], observation["q"]
                stop_rgb, stop_q = observation["stop_rgb"], observation["stop_q"]
                null_predictions = []
                if 'null_rgb' in observation and 'null_dq' in observation:
                    for null_rgb, null_dq in zip(observation['null_rgb'], observation['null_dq']):
                        null_predictions.append(infer(null_rgb, q+null_dq)[0])
            prediction, jacobian = infer(rgb, q)
            stop_prediction, _ = infer(stop_rgb, stop_q)
            # This is the only privileged read: after predictions are computed.
            with np.load(data / "evaluation" / split / path.name, allow_pickle=False) as labels:
                truth, stop_truth = labels["body_mask"].astype(bool), labels["stop_mask"].astype(bool)
                extra_labels = {key: labels[key].astype(bool) for key in (
                    'distractor_mask', 'stop_distractor_mask', 'null_body_mask', 'null_distractor_mask',
                    'unoccluded_body_mask') if key in labels}
            source_metrics = mask_metrics(prediction, truth)
            stop_metrics = mask_metrics(stop_prediction, stop_truth)
            row = {"id": path.stem, "source": source_metrics, "stopped_after_motion": stop_metrics}
            row['environment_profile'] = profiles.get(path.stem, 'clean')
            if 'unoccluded_body_mask' in extra_labels:
                unoccluded = extra_labels['unoccluded_body_mask']
                covered = int(np.sum(unoccluded & ~truth))
                row['occlusion'] = {'body_pixels_without_props': int(unoccluded.sum()),
                                    'covered_body_pixels': covered,
                                    'covered_body_fraction': covered / max(int(unoccluded.sum()), 1)}
            if 'distractor_mask' in extra_labels:
                row['distractors'] = distractor_metrics(prediction, extra_labels['distractor_mask'])
                row['stopped_distractors'] = distractor_metrics(stop_prediction, extra_labels['stop_distractor_mask'])
            if 'null_body_mask' in extra_labels:
                row['null_controls'] = []
                for null_prediction, null_truth, null_distractor in zip(
                        null_predictions, extra_labels['null_body_mask'], extra_labels['null_distractor_mask']):
                    body_score = mask_metrics(null_prediction, null_truth)
                    prop_score = distractor_metrics(null_prediction, null_distractor)
                    null_body_scores.append(body_score)
                    null_distractor_scores.append(prop_score)
                    row['null_controls'].append({'body': body_score, 'distractors': prop_score})
            if cache is not None:
                with np.load(cache / split / path.name, allow_pickle=False) as stored:
                    cached = {key: stored[key] for key in ("flow", "valid", "dq_scaled", "labels")}
                motion = _motion_scores(jacobian, cached, rng)
                partial = _label_scores(cached["labels"], truth)
                row["motion"] = motion
                row["partial_labels"] = _finish_label_scores(partial)
                motions.append(motion)
                label_rows.append(partial)
            rows.append(row)
            examples.append({"id": path.stem, "rgb": rgb, "prediction": prediction, "truth": truth,
                             "stop_rgb": stop_rgb, "stop_prediction": stop_prediction, "stop_truth": stop_truth,
                             "source_metrics": source_metrics, "stop_metrics": stop_metrics})
            print(f"{path.stem}: IoU={source_metrics['iou']:.4f}; stopped IoU={stop_metrics['iou']:.4f}", flush=True)

    metrics: dict[str, Any] = {
        "split": split, "threshold": 0.5, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "data": str(data.resolve()), "data_seed": metadata["seed"],
        "shadows": bool(metadata.get("shadows", False)),
        "scene_file": metadata.get("scene_file"), "scene_sha256": metadata.get("scene_sha256"),
        "model_config": saved["model_config"], "training": saved.get("training", {}),
        "preprocessing": saved["preprocessing"],
        "source": _aggregate_masks([row["source"] for row in rows]),
        "stopped_after_motion": _aggregate_masks([row["stopped_after_motion"] for row in rows]),
        "motion": _aggregate_motion(motions) if motions else None,
        "partial_labels": _finish_label_scores({key: sum(row[key] for row in label_rows) for key in label_rows[0]}) if label_rows else None,
        "per_pose": rows, "stop_protocol": metadata.get("stop_protocol", "not recorded"),
        "metric_conventions": {"empty_union_iou": 1, "no_prediction_with_positive_truth_precision": 0,
                               "empty_truth_recall": 1, "moving_flow_threshold_pixels": 0.35,
                               "motion_target": "RGB-estimated optical flow, not simulator flow",
                               "shuffle_scope": "negative-probe dq pairing within each initial pose"},
        "limitations": ["generic seven actuator simulator, not SO-ARM", "shadows on" if metadata.get("shadows", False) else "shadows off", "fixed camera and appearance",
                        "ideal feedback without noise or latency", "no real-hardware validation",
                        "post-motion stop frames have different poses from initial frames",
                        "negative labels include a frozen reference fitted only on training RGB",
                        "background reference assumes repeated visibility under varied poses",
                        "per-frame inference without learned temporal memory"],
    }
    if any('distractors' in row for row in rows):
        metrics['by_environment'] = {}
        for profile in sorted(set(row['environment_profile'] for row in rows)):
            group = [row for row in rows if row['environment_profile'] == profile]
            metrics['by_environment'][profile] = {
                'source': _aggregate_masks([row['source'] for row in group]),
                'stopped_after_motion': _aggregate_masks([row['stopped_after_motion'] for row in group]),
                'distractors': _aggregate_distractors([row['distractors'] for row in group]),
                'stopped_distractors': _aggregate_distractors([row['stopped_distractors'] for row in group])}
            null_group = [control for row in group for control in row.get('null_controls', [])]
            if null_group:
                metrics['by_environment'][profile]['null_controls'] = {
                    'body': _aggregate_masks([control['body'] for control in null_group]),
                    'distractors': _aggregate_distractors([control['distractors'] for control in null_group])}
        metrics['distractors'] = _aggregate_distractors([row['distractors'] for row in rows])
        metrics['environment_protocol'] = metadata.get('distractor_policy')
        if 'occlusion' in rows[0]:
            metrics['occlusion_coverage'] = {
                'poses_with_occlusion': sum(row['occlusion']['covered_body_pixels'] > 0 for row in rows),
                'mean_covered_body_fraction': float(np.mean([row['occlusion']['covered_body_fraction'] for row in rows])),
                'by_environment': {profile: {
                    'poses_with_occlusion': sum(row['occlusion']['covered_body_pixels'] > 0 for row in rows if row['environment_profile'] == profile),
                    'mean_covered_body_fraction': float(np.mean([row['occlusion']['covered_body_fraction'] for row in rows if row['environment_profile'] == profile]))}
                    for profile in metrics['by_environment']}}
        metrics['limitations'] = [item for item in metrics['limitations'] if item != 'fixed camera and appearance']
        metrics['limitations'] += ['fixed camera; kinematic noncontact distractors',
                                  'appearance varies between groups, not during signed probes',
                                  'raw image-flow EPE includes independently moving environment and is not pure self-motion error']
    if null_body_scores:
        metrics['null_controls'] = {'body': _aggregate_masks(null_body_scores),
                                    'distractors': _aggregate_distractors(null_distractor_scores)}
    if feedback_diagnostic is not None:
        diagnostic = json.loads(Path(feedback_diagnostic).read_text(encoding="utf-8"))
        metadata_hash = hashlib.sha256((data / "metadata.json").read_bytes()).hexdigest()
        if (diagnostic["checkpoint_sha256"] != metrics["checkpoint_sha256"]
                or diagnostic["metadata_sha256"] != metadata_hash
                or diagnostic["split"] != split
                or diagnostic["pose_count"] != len(files)):
            raise ValueError("Feedback diagnostic must match this checkpoint and evaluation dataset")
        metrics["current_q_perturbation"] = {
            key: diagnostic[key] for key in ("diagnostic", "summary", "protocol", "limitations")}
    image_path = output / "examples.png"
    metrics["example_pose_ids"] = _examples_image(examples, image_path)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output / "report.html").write_text(_html_report(metrics, image_path), encoding="utf-8")
    print(f"Report: {(output / 'report.html').resolve()}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--feedback-diagnostic", type=Path)
    evaluate_checkpoint(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
