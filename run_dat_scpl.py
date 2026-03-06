# run_one_ssod_full_win_v2_memsafe_autoteacher_splitfix.py
# Windows 稳定版完整 SSOD（带 cache + 可复用 fused labels）：
# mode=cache_only: Teacher(auto if missing) -> (weak/strong) pseudo -> DAT fuse  (生成 cache)
# mode=from_cache: SCPL -> pseudo dataset -> Student train -> Final val          (复用 cache)
# mode=full:       全流程（不建议 sweep 时用）
#
# 适配 4GB RTX3050：chunk + pred_batch=1 + half + nms=False + empty_cache

import os
import sys
import json
import csv
import shutil
import random
import argparse
import contextlib
import gc
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torchvision.ops import nms as tv_nms
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NAMES = ['open', 'short', 'mousebite', 'spur', 'pinhole', 'spurious_copper']


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    # 运行模式
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "cache_only", "from_cache"],
        help="full: full pipeline; cache_only: build cache (teacher+pseudo+fuse); from_cache: reuse cache -> SCPL+student+val",
    )

    p.add_argument("--tag", type=str, default="r05_seed0", help="exp tag (student/final outputs use this)")
    p.add_argument("--split_tag", type=str, default=None, help="which split yaml/images/labels to use, default uses --tag")
    p.add_argument(
        "--cache_tag",
        type=str,
        default=None,
        help="cache base tag name, e.g. r10_seed0_cache. Default: <split_tag>_cache",
    )

    p.add_argument("--root", type=str, required=True, help="deeppcb_yolo root")
    p.add_argument(
        "--ultra_root",
        type=str,
        default="",
        help="optional: ultralytics repo root to lock import (must contain ultralytics/__init__.py). If empty, use env package.",
    )
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)

    # 显存相关（4GB 建议保持默认）
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--chunk", type=int, default=8, help="4GB建议 4~16")
    p.add_argument("--pred_batch", type=int, default=1, help="必须 1 才稳")

    # 伪标签参数
    p.add_argument("--conf_weak", type=float, default=0.05)
    p.add_argument("--conf_strong", type=float, default=0.05)
    p.add_argument("--iou_nms", type=float, default=0.6)
    p.add_argument("--topk_before_nms", type=int, default=2000)
    p.add_argument("--max_det_write", type=int, default=100)

    # DAT + SCPL
    p.add_argument("--fuse_iou", type=float, default=0.6)
    p.add_argument("--scpl_base", type=float, default=0.25)
    p.add_argument("--scpl_percentile", type=int, default=60)
    p.add_argument("--scpl_max_per_image", type=int, default=80)

    # teacher 训练（cache_only/full 用；from_cache 不会训练 teacher）
    p.add_argument("--teacher_epochs", type=int, default=120)
    p.add_argument("--teacher_batch", type=int, default=2)
    p.add_argument("--teacher_model", type=str, default="yolo11n.pt")

    # student 训练（from_cache/full 用）
    p.add_argument("--student_epochs", type=int, default=120)
    p.add_argument("--train_batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--student_model", type=str, default="yolo11n.pt")

    # runs 目录（你最常用 runs_ssod_paper3_one）
    p.add_argument("--runs_dir", type=str, default="runs_ssod_paper3_one")

    return p.parse_args()


# =========================
# Utils
# =========================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(d: Path) -> List[Path]:
    out = []
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def find_split_yaml(root: Path, tag: str) -> Path:
    p = root / "splits" / tag / f"data_{tag}.yaml"
    if p.exists():
        return p
    folder = root / "splits" / tag
    ys = sorted(list(folder.glob("data_*.yaml")) + list(folder.glob("data_*.yml")))
    if len(ys) == 1:
        return ys[0]
    raise FileNotFoundError(f"Split yaml not found under {folder}")


def lock_ultralytics_import(ultra_root: str):
    """
    只在 ultra_root 确实是 ultralytics 仓库根目录（含 ultralytics/__init__.py）时才锁定。
    否则用 conda 环境里的 ultralytics，避免出现你之前看到的:
      ultralytics from: None / YOLO import error
    """
    if not ultra_root:
        import ultralytics  # noqa
        return

    ur = Path(ultra_root).resolve()
    init_py = ur / "ultralytics" / "__init__.py"
    if init_py.exists():
        sys.path.insert(0, str(ur))
        if "ultralytics" in sys.modules:
            del sys.modules["ultralytics"]
        import ultralytics  # noqa
        return
    else:
        # ultra_root 无效就不乱插 sys.path
        import ultralytics  # noqa
        return


def _xyxy_to_yolo_xywhn(xyxy: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy.unbind(1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return torch.stack([cx, cy, bw, bh], dim=1)


def _write_yolo_txt_with_conf(txt_path: Path, cls: torch.Tensor, xywhn: torch.Tensor, conf: torch.Tensor):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(len(cls)):
        c = int(cls[i].item())
        x, y, w, h = xywhn[i].tolist()
        s = float(conf[i].item())
        lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.6f}\n")
    txt_path.write_text("".join(lines), encoding="utf-8")


def read_yolo_txt_with_conf(p: Path) -> List[Tuple[int, float, float, float, float, float]]:
    if not p.exists():
        return []
    s = p.read_text(encoding="utf-8").strip()
    if not s:
        return []
    out = []
    for line in s.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        cf = float(parts[5])
        out.append((c, x, y, w, h, cf))
    return out


def xywhn_to_xyxy_pix(x: float, y: float, w: float, h: float, W: int, H: int):
    cx = x * W
    cy = y * H
    bw = w * W
    bh = h * H
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return x1, y1, x2, y2


# =========================
# Runner
# =========================
class SSODFullRunner:
    def __init__(self, args):
        self.args = args
        self.ROOT = Path(args.root).resolve()
        self.MODE = args.mode
        self.TAG = args.tag
        self.SPLIT_TAG = args.split_tag if args.split_tag else args.tag
        self.CACHE_TAG = args.cache_tag if args.cache_tag else f"{self.SPLIT_TAG}_cache"
        self.DEVICE = args.device

        self.IMGSZ = args.imgsz
        self.HALF = bool(args.half)
        self.CHUNK = int(args.chunk)
        self.PRED_BATCH = int(args.pred_batch)

        self.GEN_CONF_WEAK = float(args.conf_weak)
        self.GEN_CONF_STRONG = float(args.conf_strong)
        self.GEN_IOU_NMS = float(args.iou_nms)
        self.TOPK_BEFORE_NMS = int(args.topk_before_nms)
        self.GEN_MAX_DET_WRITE = int(args.max_det_write)

        self.FUSE_IOU = float(args.fuse_iou)
        self.SCPL_BASE = float(args.scpl_base)
        self.SCPL_PERCENTILE = int(args.scpl_percentile)
        self.SCPL_MAX_PER_IMAGE = int(args.scpl_max_per_image)

        self.TEACHER_EPOCHS = int(args.teacher_epochs)
        self.TEACHER_BATCH = int(args.teacher_batch)
        self.TEACHER_MODEL = str(args.teacher_model)

        self.STUDENT_EPOCHS = int(args.student_epochs)
        self.TRAIN_BATCH = int(args.train_batch)
        self.WORKERS = int(args.workers)
        self.STUDENT_MODEL = str(args.student_model)

        # runs dir
        self.RUNS = self.ROOT / args.runs_dir

        # ========== Cache paths (固定用 CACHE_TAG) ==========
        self.CACHE_TEACHER_ROOT = self.RUNS / f"{self.CACHE_TAG}_teacher"
        self.CACHE_TMP1_DIR = self.RUNS / f"{self.CACHE_TAG}_pseudo_tmp1"
        self.CACHE_TMP2_DIR = self.RUNS / f"{self.CACHE_TAG}_pseudo_tmp2"
        self.CACHE_FUSED_DIR = self.RUNS / f"{self.CACHE_TAG}_pseudo_fused"

        # ========== Per-exp paths (用 TAG) ==========
        self.PSEUDO_LABELS_DIR = self.RUNS / f"{self.TAG}_pseudo_labels"
        self.PSEUDO_DATASET_DIR = self.RUNS / f"{self.TAG}_pseudo_dataset"
        self.STUDENT_DIR = self.RUNS / f"{self.TAG}_student"
        self.FINAL_VAL_DIR = self.RUNS / f"{self.TAG}_final_val"

    # -------------------------
    # teacher：如果不存在就先训练（写入 cache_teacher_root）
    # -------------------------
    def ensure_teacher_exists(self, data_yaml: Path, seed: int):
        if self.CACHE_TEACHER_ROOT.exists():
            return

        print(f"⚠️ Cache teacher root not found, auto-training teacher first: {self.CACHE_TEACHER_ROOT}")
        self.CACHE_TEACHER_ROOT.mkdir(parents=True, exist_ok=True)

        from ultralytics import YOLO

        model = YOLO(self.TEACHER_MODEL)
        model.train(
            data=str(data_yaml),
            epochs=self.TEACHER_EPOCHS,
            imgsz=self.IMGSZ,
            batch=self.TEACHER_BATCH,
            workers=self.WORKERS,
            amp=True,
            deterministic=True,
            seed=seed,
            device=self.DEVICE,
            project=str(self.CACHE_TEACHER_ROOT),
            name="teacher",
            exist_ok=True,
            plots=False,
            val=True,
            conf=0.001,
            iou=0.7,
        )
        print("✅ Teacher training finished (cache).")

    def find_teacher_weights(self) -> Path:
        if not self.CACHE_TEACHER_ROOT.exists():
            raise FileNotFoundError(f"Teacher root not found: {self.CACHE_TEACHER_ROOT}")

        candidates = []
        for wdir in self.CACHE_TEACHER_ROOT.rglob("weights"):
            if (wdir / "best.pt").exists():
                candidates.append(wdir / "best.pt")
            elif (wdir / "last.pt").exists():
                candidates.append(wdir / "last.pt")

        if not candidates:
            raise FileNotFoundError(f"No teacher weights found under: {self.CACHE_TEACHER_ROOT}")

        for p in candidates:
            if p.name == "best.pt":
                print("✅ Using teacher best.pt:", p)
                return p
        print("⚠️ Using teacher last.pt:", candidates[0])
        return candidates[0]

    # -------------------------
    # 分块预测并写 txt（带 conf）
    # -------------------------
    def predict_save_txt(self, model, imgs: List[Path], out_dir: Path, augment: bool, gen_conf: float) -> Path:
        out_dir = Path(out_dir)
        pred_dir = out_dir / "pred"
        labels_dir = pred_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        log_file = pred_dir / "predict_stdout.log"
        pred_dir.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as lf, \
             contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):

            for i in range(0, len(imgs), self.CHUNK):
                sub = imgs[i:i + self.CHUNK]
                sub_str = [str(p) for p in sub]

                with torch.inference_mode():
                    gen = model.predict(
                        source=sub_str,
                        imgsz=self.IMGSZ,
                        conf=gen_conf,
                        iou=self.GEN_IOU_NMS,
                        max_det=30000,
                        save=False,
                        save_txt=False,
                        save_conf=False,
                        augment=augment,
                        project=str(out_dir),
                        name="pred",
                        exist_ok=True,
                        verbose=False,
                        stream=True,
                        batch=self.PRED_BATCH,
                        device=self.DEVICE,
                        half=self.HALF,
                        nms=False,   # ✅ 禁用内部 NMS
                    )

                    for res in gen:
                        img_path = Path(res.path)
                        txt_path = labels_dir / f"{img_path.stem}.txt"

                        if res.boxes is None or len(res.boxes) == 0:
                            txt_path.write_text("", encoding="utf-8")
                            continue

                        xyxy = res.boxes.xyxy
                        conf = res.boxes.conf
                        cls = res.boxes.cls

                        m = conf >= gen_conf
                        xyxy, conf, cls = xyxy[m], conf[m], cls[m]
                        if xyxy.numel() == 0:
                            txt_path.write_text("", encoding="utf-8")
                            continue

                        if conf.numel() > self.TOPK_BEFORE_NMS:
                            topk = torch.topk(conf, k=self.TOPK_BEFORE_NMS)
                            idx = topk.indices
                            xyxy, conf, cls = xyxy[idx], conf[idx], cls[idx]

                        keep_all = []
                        for c in cls.unique():
                            idx = (cls == c).nonzero(as_tuple=False).squeeze(1)
                            k = tv_nms(xyxy[idx], conf[idx], self.GEN_IOU_NMS)
                            keep_all.append(idx[k])
                        keep = torch.cat(keep_all, dim=0) if keep_all else torch.empty((0,), dtype=torch.long)

                        if keep.numel() == 0:
                            txt_path.write_text("", encoding="utf-8")
                            continue

                        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

                        if conf.numel() > self.GEN_MAX_DET_WRITE:
                            topk = torch.topk(conf, k=self.GEN_MAX_DET_WRITE)
                            idx = topk.indices
                            xyxy, conf, cls = xyxy[idx], conf[idx], cls[idx]

                        h0, w0 = res.orig_shape
                        xywhn = _xyxy_to_yolo_xywhn(xyxy, w0, h0).clamp(0, 1)
                        _write_yolo_txt_with_conf(
                            txt_path,
                            cls.detach().cpu(),
                            xywhn.detach().cpu(),
                            conf.detach().cpu()
                        )

                del gen
                torch.cuda.empty_cache()
                gc.collect()

        return labels_dir

    # -------------------------
    # DAT fuse (weak + strong)
    # -------------------------
    def fuse_two_labelsets(self, img_path: Path, weak_txt: Path, strong_txt: Path, out_txt: Path):
        W, H = Image.open(img_path).size

        a = read_yolo_txt_with_conf(weak_txt)
        b = read_yolo_txt_with_conf(strong_txt)
        all_boxes = a + b
        if not all_boxes:
            out_txt.write_text("", encoding="utf-8")
            return

        cls_list = torch.tensor([t[0] for t in all_boxes], dtype=torch.float32)
        conf_list = torch.tensor([t[5] for t in all_boxes], dtype=torch.float32)

        xyxy = []
        for (c, x, y, w, h, cf) in all_boxes:
            xyxy.append(xywhn_to_xyxy_pix(x, y, w, h, W, H))
        xyxy = torch.tensor(xyxy, dtype=torch.float32)

        keep_all = []
        for c in cls_list.unique():
            idx = (cls_list == c).nonzero(as_tuple=False).squeeze(1)
            k = tv_nms(xyxy[idx], conf_list[idx], self.FUSE_IOU)
            keep_all.append(idx[k])
        keep = torch.cat(keep_all, dim=0) if keep_all else torch.empty((0,), dtype=torch.long)

        if keep.numel() == 0:
            out_txt.write_text("", encoding="utf-8")
            return

        xyxy = xyxy[keep]
        conf_list = conf_list[keep]
        cls_list = cls_list[keep]

        if conf_list.numel() > self.SCPL_MAX_PER_IMAGE:
            topk = torch.topk(conf_list, k=self.SCPL_MAX_PER_IMAGE)
            idx = topk.indices
            xyxy, conf_list, cls_list = xyxy[idx], conf_list[idx], cls_list[idx]

        x1, y1, x2, y2 = xyxy.unbind(1)
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        xywhn = torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1)

        _write_yolo_txt_with_conf(out_txt, cls_list.long(), xywhn, conf_list)

    # -------------------------
    # SCPL filter（按类分位数阈值）
    # -------------------------
    def scpl_filter_labels(self, fused_labels_dir: Path, out_labels_dir: Path):
        fused_labels_dir = Path(fused_labels_dir)
        out_labels_dir = Path(out_labels_dir)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        per_cls: Dict[int, List[float]] = {}
        all_txt = sorted(fused_labels_dir.glob("*.txt"))
        for p in all_txt:
            for (c, x, y, w, h, cf) in read_yolo_txt_with_conf(p):
                per_cls.setdefault(c, []).append(cf)

        thr: Dict[int, float] = {}
        for c in range(6):
            confs = per_cls.get(c, [])
            if not confs:
                thr[c] = self.SCPL_BASE
            else:
                confs_sorted = sorted(confs)
                k = int(round((self.SCPL_PERCENTILE / 100.0) * (len(confs_sorted) - 1)))
                k = max(0, min(k, len(confs_sorted) - 1))
                thr[c] = max(self.SCPL_BASE, float(confs_sorted[k]))

        kept_total = 0
        for p in all_txt:
            rows = read_yolo_txt_with_conf(p)
            keep_rows = []
            for (c, x, y, w, h, cf) in rows:
                if cf >= thr.get(c, self.SCPL_BASE):
                    keep_rows.append((c, x, y, w, h))
            out_p = out_labels_dir / p.name
            if not keep_rows:
                out_p.write_text("", encoding="utf-8")
                continue
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for (c, x, y, w, h) in keep_rows]
            out_p.write_text("".join(lines), encoding="utf-8")
            kept_total += len(keep_rows)

        self.PSEUDO_LABELS_DIR.mkdir(parents=True, exist_ok=True)
        thr_path = self.PSEUDO_LABELS_DIR / "scpl_thresholds.json"
        thr_path.write_text(json.dumps({str(k): v for k, v in thr.items()}, indent=2), encoding="utf-8")
        print("✅ SCPL thresholds saved:", thr_path)
        print("✅ SCPL kept labels total:", kept_total)

    # -------------------------
    # build pseudo dataset
    # -------------------------
    def build_pseudo_dataset(self, labeled_img_dir: Path, unlabeled_imgs: List[Path], pseudo_labels_dir: Path) -> Path:
        ds = self.PSEUDO_DATASET_DIR
        img_out = ds / "images" / "train"
        lab_out = ds / "labels" / "train"
        img_out.mkdir(parents=True, exist_ok=True)
        lab_out.mkdir(parents=True, exist_ok=True)

        labeled_imgs = list_images(labeled_img_dir)
        labeled_labels_dir = self.ROOT / "labels" / f"train_{self.SPLIT_TAG}"
        if not labeled_labels_dir.exists():
            raise FileNotFoundError(f"Missing labeled GT labels dir: {labeled_labels_dir}")

        # labeled: GT
        for im in labeled_imgs:
            shutil.copy2(im, img_out / im.name)
            gt = labeled_labels_dir / f"{im.stem}.txt"
            if gt.exists():
                shutil.copy2(gt, lab_out / gt.name)
            else:
                (lab_out / f"{im.stem}.txt").write_text("", encoding="utf-8")

        # unlabeled: pseudo
        for im in unlabeled_imgs:
            shutil.copy2(im, img_out / im.name)
            pl = Path(pseudo_labels_dir) / f"{im.stem}.txt"
            if pl.exists():
                shutil.copy2(pl, lab_out / pl.name)
            else:
                (lab_out / f"{im.stem}.txt").write_text("", encoding="utf-8")

        out_yaml = ds / "data.yaml"
        text = (
            f"path: {ds.as_posix()}\n"
            f"train: images/train\n"
            f"val: {(self.ROOT / 'images' / 'val').as_posix()}\n"
            f"test: {(self.ROOT / 'images' / 'test').as_posix()}\n"
            f"nc: 6\n"
            f"names: {NAMES}\n"
        )
        out_yaml.write_text(text, encoding="utf-8")
        print("✅ pseudo dataset yaml:", out_yaml)
        return out_yaml

    # -------------------------
    # train student
    # -------------------------
    def train_student(self, pseudo_yaml: Path, seed: int):
        from ultralytics import YOLO

        self.STUDENT_DIR.mkdir(parents=True, exist_ok=True)
        model = YOLO(self.STUDENT_MODEL)
        model.train(
            data=str(pseudo_yaml),
            epochs=self.STUDENT_EPOCHS,
            imgsz=self.IMGSZ,
            batch=self.TRAIN_BATCH,
            workers=self.WORKERS,
            amp=True,
            deterministic=True,
            seed=seed,
            device=self.DEVICE,
            project=str(self.STUDENT_DIR),
            name="student",
            plots=True,
            val=True,
            conf=0.001,
            iou=0.7,
        )

    def resolve_student_weights(self) -> Path:
        candidates = []
        for wdir in self.STUDENT_DIR.rglob("weights"):
            if (wdir / "best.pt").exists():
                candidates.append(wdir / "best.pt")
            elif (wdir / "last.pt").exists():
                candidates.append(wdir / "last.pt")
        if not candidates:
            raise FileNotFoundError(f"No student weights found under: {self.STUDENT_DIR}")
        for p in candidates:
            if p.name == "best.pt":
                return p
        return candidates[0]

    # -------------------------
    # final val
    # -------------------------
    def final_val(self, model_pt: Path, data_yaml: Path):
        from ultralytics import YOLO

        self.FINAL_VAL_DIR.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_pt))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=self.IMGSZ,
            batch=4,
            workers=self.WORKERS,
            conf=0.01,
            iou=0.7,
            device=self.DEVICE,
        )

        out_csv = self.FINAL_VAL_DIR / "final_metrics.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["exp", "mAP50", "mAP50-95", "P", "R"])
            w.writerow([self.TAG, float(metrics.box.map50), float(metrics.box.map), float(metrics.box.mp), float(metrics.box.mr)])
        print("✅ final val saved:", out_csv)

    # -------------------------
    # cache build: teacher + pseudo + fuse (写入 CACHE_TAG)
    # -------------------------
    def build_cache_if_needed(self, data_yaml: Path, seed: int, unlabeled_imgs: List[Path]):
        from ultralytics import YOLO

        # 1) teacher
        self.ensure_teacher_exists(data_yaml=data_yaml, seed=seed)
        teacher_pt = self.find_teacher_weights()
        teacher = YOLO(str(teacher_pt))

        # 2) pseudo weak/strong：如果目录存在且 labels 数量对得上，就跳过
        labels1_dir = self.CACHE_TMP1_DIR / "pred" / "labels"
        labels2_dir = self.CACHE_TMP2_DIR / "pred" / "labels"

        def _looks_done(lbl_dir: Path) -> bool:
            if not lbl_dir.exists():
                return False
            # 允许少量空 txt，但至少应该有“接近” unlabeled 数量的文件
            cnt = len(list(lbl_dir.glob("*.txt")))
            return cnt >= max(1, int(0.95 * len(unlabeled_imgs)))

        if _looks_done(labels1_dir):
            print("✅ cache weak pseudo exists, skip:", labels1_dir)
        else:
            print("==> Build cache: Pseudo weak (chunked)...")
            self.predict_save_txt(teacher, unlabeled_imgs, self.CACHE_TMP1_DIR, augment=False, gen_conf=self.GEN_CONF_WEAK)

        if _looks_done(labels2_dir):
            print("✅ cache strong pseudo exists, skip:", labels2_dir)
        else:
            print("==> Build cache: Pseudo strong (chunked)...")
            self.predict_save_txt(teacher, unlabeled_imgs, self.CACHE_TMP2_DIR, augment=True, gen_conf=self.GEN_CONF_STRONG)

        # 3) fuse：如果 fused labels 已有且数量够，就跳过
        fused_labels_dir = self.CACHE_FUSED_DIR / "labels"
        fused_labels_dir.mkdir(parents=True, exist_ok=True)

        fused_done = len(list(fused_labels_dir.glob("*.txt"))) >= max(1, int(0.95 * len(unlabeled_imgs)))
        if fused_done:
            print("✅ cache fused labels exist, skip:", fused_labels_dir)
            return

        print("==> Build cache: DAT fuse...")
        for im in unlabeled_imgs:
            wtxt = labels1_dir / f"{im.stem}.txt"
            stxt = labels2_dir / f"{im.stem}.txt"
            out = fused_labels_dir / f"{im.stem}.txt"
            self.fuse_two_labelsets(im, wtxt, stxt, out)
        print("✅ cache fused labels saved:", fused_labels_dir)

    # -------------------------
    # main run
    # -------------------------
    def run(self):
        # seed: 支持 tag 里有后缀，如 r10_seed0_q45
        m = re.search(r"_seed(\d+)", self.TAG)
        seed = int(m.group(1)) if m else 0

        seed_everything(seed)
        print(f"MODE={self.MODE} | TAG={self.TAG} | split_tag={self.SPLIT_TAG} | cache_tag={self.CACHE_TAG} | seed={seed} | device={self.DEVICE}")
        print("RUNS DIR:", self.RUNS)

        data_yaml = find_split_yaml(self.ROOT, self.SPLIT_TAG)
        print("📄 split yaml:", data_yaml)

        labeled_img_dir = self.ROOT / "images" / f"train_{self.SPLIT_TAG}"
        full_train_dir = self.ROOT / "images" / "train"
        if not labeled_img_dir.exists():
            raise FileNotFoundError(f"Missing labeled image dir: {labeled_img_dir}")
        if not full_train_dir.exists():
            raise FileNotFoundError(f"Missing full train dir: {full_train_dir}")

        labeled_imgs = {p.name for p in list_images(labeled_img_dir)}
        all_train_imgs = list_images(full_train_dir)
        unlabeled_imgs = [p for p in all_train_imgs if p.name not in labeled_imgs]
        print("labeled:", len(labeled_imgs), "| unlabeled:", len(unlabeled_imgs))

        # ========== 1) cache_only / full: build cache ==========
        if self.MODE in ["cache_only", "full"]:
            self.build_cache_if_needed(data_yaml=data_yaml, seed=seed, unlabeled_imgs=unlabeled_imgs)
            if self.MODE == "cache_only":
                print("🎉 Cache stage finished:", self.CACHE_TAG)
                return

        # ========== 2) from_cache / full: SCPL -> student -> final val ==========
        fused_labels_dir = (self.CACHE_FUSED_DIR / "labels")
        if not fused_labels_dir.exists():
            raise FileNotFoundError(
                f"Cache fused labels not found: {fused_labels_dir}\n"
                f"Run with --mode cache_only first."
            )

        print("==> SCPL filter (from cache fused labels)...")
        final_labels_dir = self.PSEUDO_LABELS_DIR / "labels_train"
        self.scpl_filter_labels(fused_labels_dir, final_labels_dir)
        print("✅ final pseudo labels:", final_labels_dir)

        print("==> Build pseudo dataset...")
        pseudo_yaml = self.build_pseudo_dataset(labeled_img_dir, unlabeled_imgs, final_labels_dir)

        print("==> Train student...")
        self.train_student(pseudo_yaml, seed=seed)

        print("==> Final val...")
        student_pt = self.resolve_student_weights()
        self.final_val(student_pt, data_yaml)

        print("🎉 DONE:", self.TAG)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()

    lock_ultralytics_import(args.ultra_root)
    import ultralytics  # noqa
    print("✅ ultralytics from:", getattr(ultralytics, "__file__", None))
    # YOLO 也顺便试一下，避免你再踩 ImportError
    from ultralytics import YOLO  # noqa
    print("✅ YOLO import OK:", YOLO)

    runner = SSODFullRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
