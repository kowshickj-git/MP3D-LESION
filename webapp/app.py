"""Web front end for the MP3D universal lesion detector.

    .venv/Scripts/python.exe webapp/app.py     ->  http://127.0.0.1:5000

Routes
    /                overview: all eight body regions
    /r/<region>      one region, with its own samples
    /api/regions     region metadata + sample listing
    /api/detect      multipart: image, score_thr, show_gt, spacing

What the model does and does not do
    MP3D is a single-class detector: every output box is just "Lesion". It
    does NOT predict which organ or region a finding belongs to, and it does
    not diagnose. The region shown on a page is ground-truth metadata from
    DeepLesion's DL_info.csv for that sample, never a model prediction.

    The size reported per detection IS derived from the model: the box is
    converted to millimetres using the slice's pixel spacing, and the band is
    a threshold on that measurement (see SIZE_BANDS).

3D context
    MP3D consumes 9 consecutive slices. When a filename identifies a study on
    disk the real neighbours are loaded; otherwise the slice is replicated 9x
    and the UI says so.
"""
import base64
import csv
import json
import os
import os.path as osp
import re
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import mmcv
import numpy as np
import torch
from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)
from mmcv.parallel import DataContainer, collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.datasets.pipelines.image_io_3DCE import load_multislice_gray_png_3DCE
from mmdet.models import build_detector

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
CONFIG = osp.join(ROOT, 'configs/deeplesion/mp3d_groupconv_local_hires.py')
CHECKPOINT = osp.join(ROOT, 'work_dirs/mp3d_lesion_hires/epoch_7.pth')
IMG_ROOT = osp.join(ROOT, 'data/DeepLesion/Images_png/Images_png')
SAMPLES = osp.join(ROOT, 'sample_images')
ANN_FILE = osp.join(ROOT, 'data/DeepLesion/annotation/deeplesion_test_local.json')
DL_INFO = osp.join(ROOT, 'data/DeepLesion/DL_info.csv')
WINDOW = [-1024, 1050]
NUM_SLICE = 9
DEFAULT_SPACING = 0.7   # mm/px, only used when a slice has no recorded spacing

# The eight coarse regions DeepLesion labels. Text describes what radiologists
# typically bookmark there - it is context for the reader, not a model output.
REGIONS = [
    dict(
        key='lung', name='Lung',
        blurb='Pulmonary nodules and masses in the lung parenchyma.',
        pathology='A focal opacity inside aerated lung. Below 30 mm it is '
        'called a nodule, above that a mass. The common causes split three '
        'ways: primary lung cancer (adenocarcinoma, squamous cell); '
        'metastases from elsewhere, since the lungs are the single most '
        'frequent site secondary disease lands; and benign scarring such as '
        'old granulomas from tuberculosis or histoplasmosis, and hamartomas.',
        why='Nodules drive two separate workflows. Newly found ones are '
        'triaged by the Fleischner criteria, where size decides whether the '
        'patient gets a repeat scan in 3 months, in 12 months, or none at '
        'all. Known cancers are tracked by RECIST, where the long axis of '
        'the same lesion is remeasured on each scan to judge whether '
        'treatment is working.',
        ct='The easiest region in the body for a detector. Air in the lung '
        'sits near -1000 HU and a soft-tissue nodule near +30 HU, so the '
        'contrast across the lesion edge is enormous. The real difficulty is '
        'telling a nodule from a vessel seen end-on, which is exactly what '
        'the 9-slice context resolves: a vessel continues into the '
        'neighbouring slices, a nodule does not.',
        perf='3 of 3 lesions matched, our joint-best region.'),

    dict(
        key='mediastinum', name='Mediastinum',
        blurb='Lymph nodes between the lungs, around the heart and great '
        'vessels.',
        pathology='Almost always lymph nodes. They enlarge when tumour '
        'spreads into them - nodal spread from lung cancer is what sets the '
        'N stage and often decides whether surgery is still on the table - '
        'and also in lymphoma, in sarcoidosis, and in ordinary reactive '
        'infection.',
        why='A node is called pathological on its short axis, not its long: '
        '10 mm is the usual cut-off, and RECIST asks for 15 mm before a node '
        'can serve as a target lesion. That is why this app reports both '
        'axes rather than a single number.',
        ct='Moderate contrast. Nodes are soft tissue sitting in mediastinal '
        'fat, which separates them well, but they abut vessels of identical '
        'density. Without intravenous contrast a node and a branch of the '
        'pulmonary artery look much the same on one slice - again, '
        'through-plane context is what separates them.',
        perf='3 of 3 lesions matched, our joint-best region. It also has the '
        'most local training data of any region, at 143 lesions.'),

    dict(
        key='liver', name='Liver',
        blurb='Focal hepatic lesions and perihepatic nodes.',
        pathology='Extremely mixed, which is what makes the liver hard. '
        'Simple cysts and haemangiomas are common and harmless. Focal '
        'nodular hyperplasia and adenomas are benign but matter. '
        'Hepatocellular carcinoma arises in cirrhotic livers. And the liver '
        'is the dominant landing site for metastases from colorectal, '
        'pancreatic, gastric and breast primaries.',
        why='Counting and measuring liver deposits changes staging directly, '
        'and in colorectal cancer it decides whether a patient is a '
        'candidate for liver resection at all.',
        ct='The hardest region here. Liver parenchyma sits around 50-60 HU '
        'and many lesions differ from it by only 10-20 HU, so on a '
        'non-contrast scan the lesion edge can be nearly invisible. '
        'Detectability depends heavily on contrast timing, and DeepLesion '
        'mixes scan phases freely.',
        perf='0 of 4 lesions matched - our weakest region by a wide margin. '
        'Two causes compound: the low intrinsic contrast above, and only 64 '
        'liver lesions in our local training subset against 143 for '
        'mediastinum.'),

    dict(
        key='kidney', name='Kidney',
        blurb='Renal lesions, including cystic and solid masses.',
        pathology='Simple cysts are found incidentally in a large fraction '
        'of adults over 50 and are benign. The clinical question is almost '
        'always whether a lesion is a simple cyst, a complex cyst, or a '
        'solid renal cell carcinoma. Angiomyolipomas, which contain fat, and '
        'metastases make up most of the rest.',
        why='Complex renal cysts are sorted by the Bosniak classification, '
        'which decides between ignoring the lesion, imaging follow-up, and '
        'surgery. Size and growth over time are central to that call.',
        ct='Mid-difficulty. A simple cyst is water density, roughly 0-20 HU, '
        'and stands out against the ~30-40 HU cortex. Solid masses enhance '
        'with contrast but can be nearly isodense without it. The kidneys '
        'also move noticeably with respiration between scans.',
        perf='1 of 3 lesions matched, from 74 local training lesions.'),

    dict(
        key='abdomen', name='Abdomen',
        blurb='Abdominal lymph nodes and soft-tissue masses outside the '
        'solid organs.',
        pathology='A catch-all for disease that is not in a named organ: '
        'retroperitoneal and mesenteric lymph nodes, peritoneal deposits, '
        'omental cake from ovarian or gastric primaries, and masses arising '
        'from the bowel wall.',
        why='Retroperitoneal nodal disease and peritoneal spread both change '
        'staging and frequently rule out curative surgery.',
        ct='Crowded and low-contrast. Bowel loops, unopacified vessels, '
        'mesenteric fat and nodes all overlap in density, and a collapsed '
        'loop of small bowel can look very much like a nodal mass on a '
        'single slice. This is the region where a detector produces its most '
        'plausible false positives.',
        perf='1 of 3 lesions matched, despite this region having the most '
        'local training data of all - 188 lesions. The bottleneck here is '
        'the anatomy, not the sample count.'),

    dict(
        key='pelvis', name='Pelvis',
        blurb='Pelvic nodes and masses.',
        pathology='Iliac and obturator lymph nodes, adnexal and ovarian '
        'masses, and disease arising from the bladder, prostate or rectum.',
        why='Pelvic nodal status is central to staging urological, '
        'gynaecological and rectal cancers, and often determines the '
        'radiotherapy field.',
        ct='Mixed. The bony pelvis gives strong landmarks, but the bladder, '
        'rectum and uterus vary enormously in filling and position between '
        'scans, and unopacified bowel in the pelvis is a frequent mimic.',
        perf='2 of 3 lesions matched, from 61 local training lesions.'),

    dict(
        key='bone', name='Bone',
        blurb='Lytic and blastic osseous lesions.',
        pathology='Overwhelmingly metastatic. Lytic deposits, where bone is '
        'destroyed, are typical of renal, thyroid and lung primaries and of '
        'myeloma. Blastic deposits, where dense abnormal bone is laid down, '
        'are typical of prostate and some breast primaries. Primary bone '
        'tumours are rare by comparison.',
        why='Bone metastases establish metastatic disease, and they carry '
        'their own complications - fracture risk and spinal cord '
        'compression - that are managed off the imaging.',
        ct='Split personality. Blastic lesions are the highest-contrast '
        'finding in the body, bright white against marrow fat. Lytic lesions '
        'are the opposite: subtle marrow replacement that can be almost '
        'invisible until the cortex is breached. A soft-tissue window hides '
        'much of both, and the single window this model uses is a real '
        'limitation.',
        perf='1 of 3 lesions matched, from just 15 local training lesions - '
        'the thinnest data of any region here.'),

    dict(
        key='soft-tissue', name='Soft tissue',
        blurb='Masses in muscle, fat and the body wall.',
        pathology='Soft-tissue sarcomas, metastatic deposits within muscle, '
        'benign lipomas, desmoid tumours, and non-neoplastic collections '
        'such as abscesses and haematomas that can look identical to tumour '
        'on a single unenhanced slice.',
        why='Size at presentation and depth relative to the fascia are the '
        'main determinants of how a soft-tissue mass is worked up.',
        ct='Moderate. Fat around a mass provides good contrast, but a mass '
        'sitting inside muscle is nearly isodense with it, and the body wall '
        'is a region radiologists themselves scrutinise less closely.',
        perf='2 of 3 lesions matched, from 66 local training lesions.'),
]
REGION_BY_KEY = {r['key']: r for r in REGIONS}

# Long-axis size bands. RECIST treats >=10 mm as a measurable target lesion and
# >=15 mm short axis as a pathological node, which is where the first two
# thresholds come from; the top band just flags bulky disease.
SIZE_BANDS = [
    (10.0, 'sub-centimetre', 'Below the 10 mm RECIST measurable threshold.'),
    (20.0, 'small', 'Measurable, in the 10-20 mm range.'),
    (30.0, 'moderate', 'Measurable, 20-30 mm.'),
    (float('inf'), 'large', '30 mm or greater.'),
]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

MODEL = None
CFG = None
POST_PIPELINE = None
SLICE_INTV = {}
GT_BY_REL = {}
SPACING_BY_REL = {}
SAMPLE_INDEX = {}


def load_model():
    global MODEL, CFG, POST_PIPELINE
    cfg = mmcv.Config.fromfile(CONFIG)
    for key in ('backbone', 'neck', 'rpn_head'):
        if key in cfg.model and 'init_cfg' in cfg.model[key]:
            cfg.model[key].init_cfg = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    from mmcv.runner import load_checkpoint
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model.CLASSES = ('Lesion', )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    MODEL, CFG = model, cfg
    POST_PIPELINE = Compose(cfg.data.test.pipeline[1:])   # skip the file loader
    print('model ready on ' + device, flush=True)
    return device


def load_metadata():
    """Ground-truth boxes, slice intervals and pixel spacing."""
    if osp.exists(ANN_FILE):
        data = json.load(open(ANN_FILE))
        by_id = {}
        for a in data['annotations']:
            by_id.setdefault(a['image_id'], []).append(a['bbox'])
        for im in data['images']:
            SLICE_INTV[im['file_name']] = im.get('slice_intv', 2.5)
            GT_BY_REL[im['file_name']] = by_id.get(im['id'], [])
            if im.get('spacing'):
                SPACING_BY_REL[im['file_name']] = im['spacing']

    # DL_info covers every slice, including ones outside our test subset
    if osp.exists(DL_INFO):
        for r in csv.DictReader(open(DL_INFO)):
            parts = r['File_name'].split('_')
            rel = '%s/%s' % ('_'.join(parts[:3]), parts[3])
            SPACING_BY_REL.setdefault(
                rel, float(r['Spacing_mm_px_'].split(',')[0]))
            SLICE_INTV.setdefault(
                rel, float(r['Spacing_mm_px_'].split(',')[2]))

    path = osp.join(SAMPLES, 'regions.json')
    if osp.exists(path):
        for entry in json.load(open(path)):
            SAMPLE_INDEX[entry['file']] = entry


def window_hu(stack):
    im = stack.astype(np.float32) - 32768.0
    lo, hi = WINDOW
    return np.clip((im - lo) / (hi - lo), 0, 1) * 255.0


def resolve_study(filename):
    """Map an uploaded name onto a dataset slice, ignoring any label prefix."""
    stem = osp.splitext(osp.basename(filename))[0]
    m = re.search(r'(\d{6}_\d{2}_\d{2})_{1,2}(\d{1,3})$', stem)
    if not m:
        return None
    rel = '%s/%03d.png' % (m.group(1), int(m.group(2)))
    return rel if osp.exists(osp.join(IMG_ROOT, rel)) else None


def build_stack(file_bytes, filename):
    rel = resolve_study(filename)
    if rel is not None:
        intv = SLICE_INTV.get(rel, 2.5)
        arr = load_multislice_gray_png_3DCE(
            osp.join(IMG_ROOT, rel), NUM_SLICE, intv, WINDOW, False, None)
        note = ('Loaded %d real neighbouring slices from study %s - the full '
                '3D context the model was trained with.'
                % (NUM_SLICE, rel.split('/')[0]))
        return arr, 'real', note

    raw = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), -1)
    if raw is None:
        raise ValueError('Could not decode that file as an image.')
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    stacked = np.repeat(raw[:, :, None].astype(np.float32), NUM_SLICE, axis=2)
    if raw.dtype == np.uint16:
        arr = window_hu(stacked)
        note = ('16-bit CT slice, but its neighbours are not on disk, so the '
                'slice was replicated %d times. Expect more false positives '
                'than the quoted benchmark.' % NUM_SLICE)
    else:
        arr = stacked
        note = ('8-bit image: assumed already windowed and replicated %d '
                'times. No through-plane context, so treat the output as '
                'indicative only.' % NUM_SLICE)
    return arr, 'replicated', note


def iou_xyxy_xywh(det, gt):
    """IoU between a detection [x1,y1,x2,y2] and a ground truth [x,y,w,h]."""
    gx1, gy1, gw, gh = gt
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(det[0], gx1), max(det[1], gy1)
    ix2, iy2 = min(det[2], gx2), min(det[3], gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (det[2] - det[0]) * (det[3] - det[1]) + gw * gh - inter
    return inter / union if union > 0 else 0.0


def size_band(long_mm):
    for limit, label, why in SIZE_BANDS:
        if long_mm < limit:
            return label, why
    return SIZE_BANDS[-1][1], SIZE_BANDS[-1][2]


def to_png_b64(bgr):
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise ValueError('failed to encode result image')
    return 'data:image/png;base64,' + base64.b64encode(buf).decode()


def run_detection(arr, score_thr):
    device = next(MODEL.parameters()).device
    data = dict(filename='upload', ori_filename='upload', img=arr,
                img_shape=arr.shape, ori_shape=arr.shape, img_fields=['img'])
    data = POST_PIPELINE(data)
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [m.data[0] if isinstance(m, DataContainer) else m
                         for m in data['img_metas']]
    data['img'] = [i.data[0] if isinstance(i, DataContainer) else i
                   for i in data['img']]
    if device.type != 'cpu':
        data = scatter(data, [device])[0]
    started = time.time()
    with torch.no_grad():
        result = MODEL(return_loss=False, rescale=True, **data)
    elapsed = time.time() - started
    dets = result[0][0]
    return [d for d in dets if d[4] >= score_thr], float(elapsed), int(len(dets))


def render(arr, dets, gts):
    centre = arr[:, :, NUM_SLICE // 2]
    img = cv2.cvtColor(np.clip(centre, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    for x, y, w, h in gts:
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (90, 220, 90), 1)
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2, s = d
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (60, 60, 255), 2)
        cv2.putText(img, '%d' % i, (int(x1) + 3, max(13, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 255), 2,
                    cv2.LINE_AA)
    return img


def samples_for(region_key=None):
    out = []
    for name in sorted(os.listdir(SAMPLES)) if osp.isdir(SAMPLES) else []:
        if not name.lower().endswith('.png'):
            continue
        meta = SAMPLE_INDEX.get(name, {})
        if region_key and meta.get('region') != region_key:
            continue
        rel = resolve_study(name)
        out.append(dict(
            file=name,
            region=meta.get('region'),
            kind=meta.get('kind', 'lesion'),
            source=meta.get('source', rel or name),
            long_mm=meta.get('long_mm'),
            short_mm=meta.get('short_mm'),
            lesions=len(GT_BY_REL.get(rel, [])) if rel else 0,
        ))
    # lesion cases first, largest first, negative control last
    out.sort(key=lambda s: (s['kind'] == 'normal', -(s['long_mm'] or 0)))
    return out


@app.route('/')
def index():
    counts = {}
    for s in samples_for():
        counts.setdefault(s['region'], {'lesion': 0, 'normal': 0})
        counts[s['region']][s['kind']] = counts[s['region']].get(s['kind'], 0) + 1
    return render_template('index.html', regions=REGIONS, counts=counts,
                           region=None)


@app.route('/r/<region_key>')
def region_page(region_key):
    region = REGION_BY_KEY.get(region_key)
    if region is None:
        abort(404)
    return render_template('index.html', regions=REGIONS, counts={},
                           region=region)


@app.route('/samples/<path:name>')
def sample_file(name):
    return send_from_directory(SAMPLES, name)


@app.route('/api/regions')
def api_regions():
    return jsonify(regions=REGIONS, samples=samples_for())


@app.route('/api/samples')
def api_samples():
    return jsonify(samples_for(request.args.get('region')))


@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'image' not in request.files:
        return jsonify(error='No file was uploaded.'), 400
    f = request.files['image']
    if not f.filename:
        return jsonify(error='No file was selected.'), 400
    try:
        score_thr = float(request.form.get('score_thr', 0.155))
    except ValueError:
        score_thr = 0.155
    show_gt = request.form.get('show_gt', '1') == '1'

    try:
        arr, mode, note = build_stack(f.read(), f.filename)
    except Exception as exc:
        return jsonify(error=str(exc)), 400

    dets, elapsed, total = run_detection(arr, score_thr)
    rel = resolve_study(f.filename)
    gts = GT_BY_REL.get(rel, []) if (show_gt and rel) else []
    img = render(arr, dets, gts)

    spacing = SPACING_BY_REL.get(rel) if rel else None
    try:
        spacing = float(request.form.get('spacing') or spacing or DEFAULT_SPACING)
    except ValueError:
        spacing = spacing or DEFAULT_SPACING
    spacing_known = rel in SPACING_BY_REL if rel else False

    meta = SAMPLE_INDEX.get(osp.basename(f.filename), {})
    out = []
    for d in dets:
        x1, y1, x2, y2, s = [float(v) for v in d]
        w_mm, h_mm = (x2 - x1) * spacing, (y2 - y1) * spacing
        long_mm, short_mm = max(w_mm, h_mm), min(w_mm, h_mm)
        band, why = size_band(long_mm)
        best_iou = max((iou_xyxy_xywh([x1, y1, x2, y2], g) for g in gts),
                       default=0.0)
        out.append(dict(
            x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
            score=round(s, 4),
            width=round(x2 - x1, 1), height=round(y2 - y1, 1),
            long_mm=round(long_mm, 1), short_mm=round(short_mm, 1),
            band=band, band_why=why,
            iou=round(best_iou, 3), hit=bool(best_iou >= 0.5)))

    return jsonify(
        image=to_png_b64(img), mode=mode, note=note,
        elapsed_ms=round(elapsed * 1000), total_candidates=total,
        threshold=score_thr, ground_truth=len(gts),
        spacing=round(spacing, 4), spacing_known=spacing_known,
        region=meta.get('region'), kind=meta.get('kind'),
        truth_long_mm=meta.get('long_mm'), truth_short_mm=meta.get('short_mm'),
        hits=sum(1 for d in out if d['hit']),
        detections=out)


if __name__ == '__main__':
    load_metadata()
    load_model()
    print('open http://127.0.0.1:5000', flush=True)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=False)
