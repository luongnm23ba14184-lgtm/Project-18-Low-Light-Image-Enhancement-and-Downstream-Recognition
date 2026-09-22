"""
Visualization and Comparative Plotting Module.
Generates side-by-side qualitative comparisons and scientific charts for Project 18.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bounding_boxes(image_rgb: np.ndarray, boxes, class_names: dict, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws bounding boxes and labels on an image.
    boxes: Ultralytics boxes object or list of [x1, y1, x2, y2, conf, cls]
    """
    img = image_rgb.copy()
    if boxes is None:
        return img

    for box in boxes:
        # Check type
        if hasattr(box, 'xyxy'):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
        else:
            xyxy = np.array(box[:4], dtype=int)
            conf = float(box[4]) if len(box) > 4 else 1.0
            cls_id = int(box[5]) if len(box) > 5 else 0

        x1, y1, x2, y2 = xyxy
        name = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
        label = f"{name} {conf:.2f}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # Draw text label with background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - 20, 0)), (x1 + w, max(y1, 20)), color, -1)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def plot_side_by_side_comparison(
    raw_dark_rgb: np.ndarray,
    clahe_rgb: np.ndarray,
    zerodce_rgb: np.ndarray,
    detection_rgb: np.ndarray,
    save_path: str | Path | None = "Results/figures/enhancement_comparison.png",
    title: str = "Comparative Evaluation Across Pipeline Stages"
):
    """
    Plots and saves a 4-panel side-by-side comparison:
    [Raw Dark Image] | [DIP: CLAHE + Bilateral] | [Zero-DCE Enhancement] | [YOLOv8 Detection]
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    panels = [
        ("Raw Dark Image (Baseline)", raw_dark_rgb),
        ("DIP Baseline (CLAHE + Bilateral)", clahe_rgb),
        ("Deep Learning (Zero-DCE)", zerodce_rgb),
        ("Downstream Detection (YOLOv8)", detection_rgb),
    ]

    for ax, (panel_title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(
    scenarios_data: list[dict],
    save_path: str | Path | None = "Results/figures/map_comparison.png"
):
    """
    Plots a scientific bar chart comparing mAP@0.5 and mAP@0.5:0.95 across experimental scenarios.
    scenarios_data: list of dicts with keys: 'name', 'map50', 'map50_95'
    """
    names = [s['name'] for s in scenarios_data]
    map50 = [s.get('map50', 0.0) for s in scenarios_data]
    map50_95 = [s.get('map50_95', 0.0) for s in scenarios_data]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, map50, width, label='mAP@0.5', color='#1f77b4', edgecolor='black')
    rects2 = ax.bar(x + width / 2, map50_95, width, label='mAP@0.5:0.95', color='#ff7f0e', edgecolor='black')

    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
    ax.set_title('Experimental Evaluation Across 4 Pipeline Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.0)

    # Attach value labels
    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()
