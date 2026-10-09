"""Run MP3D lesion detection on DeepLesion CT slices and save visualisations.

Usage:
  python tools/lesion_demo.py CONFIG CHECKPOINT --ann data/DeepLesion/annotation/deeplesion_test_local.json -n 8
  python tools/lesion_demo.py CONFIG CHECKPOINT --img 000001_01_01/109.png --slice-intv 5.0
"""
import argparse
import json
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer, collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

IMG_ROOT = 'data/DeepLesion/Images_png/Images_png'


def build_model(cfg_path, ckpt, device):
    cfg = mmcv.Config.fromfile(cfg_path)
    if ckpt:
        # a fine-tuned checkpoint supersedes the COCO pretrained init
        for k in ('backbone', 'neck', 'rpn_head'):
            if k in cfg.model and 'init_cfg' in cfg.model[k]:
                cfg.model[k].init_cfg = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if ckpt:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, ckpt, map_location='cpu')
    else:
        model.init_weights()
    model.CLASSES = ('Lesion', )
    model.cfg = cfg
    model.to(device).eval()
    return model, cfg


def run_one(model, cfg, filename, slice_intv, device):
    pipeline = Compose(cfg.data.test.pipeline)
    data = dict(
        img_info=dict(filename=filename, slice_intv=slice_intv),
        img_prefix=IMG_ROOT)
    data = pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # the lesion test pipeline uses ImageToTensor (not DefaultFormatBundle), so
    # 'img' comes out as an already-batched (1, 1, D, H, W) tensor rather than a
    # DataContainer; only img_metas needs unwrapping.
    data['img_metas'] = [m.data[0] if isinstance(m, DataContainer) else m
                         for m in data['img_metas']]
    data['img'] = [i.data[0] if isinstance(i, DataContainer) else i
                   for i in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result[0][0]  # single class -> (N, 5) array of [x1,y1,x2,y2,score]


def ct_to_bgr(path, window=(-1024, 1050)):
    """Load a 16-bit DeepLesion png and window it to an 8-bit BGR image."""
    im = cv2.imread(path, -1).astype(np.float32) - 32768
    im = np.clip((im - window[0]) / (window[1] - window[0]), 0, 1) * 255
    return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def draw(path, dets, gts, out_path, score_thr):
    img = ct_to_bgr(path)
    for g in gts:
        x, y, w, h = g
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 200, 0), 1)
    n = 0
    for det in dets:
        x1, y1, x2, y2, s = det
        if s < score_thr:
            continue
        n += 1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)
        cv2.putText(img, '%.2f' % s, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(img, 'green=ground truth  red=MP3D detection', (8, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    mmcv.mkdir_or_exist(osp.dirname(out_path))
    cv2.imwrite(out_path, img)
    return n


def iou(box, gt):
    gx1, gy1, gw, gh = gt
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(box[0], gx1), max(box[1], gy1)
    ix2, iy2 = min(box[2], gx2), min(box[3], gy2)
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    union = (box[2] - box[0]) * (box[3] - box[1]) + gw * gh - inter
    return inter / union if union > 0 else 0.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint', nargs='?', default=None)
    p.add_argument('--ann', default=None, help='pick images from this json')
    p.add_argument('--img', default=None, help='single relative png path')
    p.add_argument('--slice-intv', type=float, default=2.5)
    p.add_argument('-n', '--num', type=int, default=8)
    p.add_argument('--score-thr', type=float, default=0.5)
    p.add_argument('--out', default='work_dirs/demo')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, cfg = build_model(args.config, args.checkpoint, device)

    targets = []
    if args.img:
        targets.append((args.img, args.slice_intv, []))
    else:
        data = json.load(open(args.ann))
        gt_by_img = {}
        for a in data['annotations']:
            gt_by_img.setdefault(a['image_id'], []).append(a['bbox'])
        for im in data['images'][:args.num]:
            targets.append((im['file_name'], im.get('slice_intv', 2.5),
                            gt_by_img.get(im['id'], [])))

    hits = total_gt = 0
    for fn, intv, gts in targets:
        dets = run_one(model, cfg, fn, intv, device)
        out_path = osp.join(args.out, fn.replace('/', '_'))
        n = draw(osp.join(IMG_ROOT, fn), dets, gts, out_path, args.score_thr)
        kept = dets[dets[:, 4] >= args.score_thr] if len(dets) else dets
        hit = sum(1 for g in gts
                  if any(iou(b, g) >= 0.5 for b in kept))
        hits += hit
        total_gt += len(gts)
        top = dets[0][4] if len(dets) else 0.
        print('%-24s dets>=%.2f: %2d | gt: %d | matched: %d | top score %.3f -> %s'
              % (fn, args.score_thr, n, len(gts), hit, top, out_path))
    if total_gt:
        print('\nlesions hit at IoU 0.5: %d/%d (%.1f%%)'
              % (hits, total_gt, 100. * hits / total_gt))


if __name__ == '__main__':
    main()
