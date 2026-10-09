"""Build per-body-region sample sets for the web front end.

DeepLesion's coarse lesion type lives in DL_info.csv, not in the COCO json we
train from, so this joins the two. For each of the 8 regions it exports a few
lesion slices plus one lesion-free slice from the same study (same anatomy,
far from any bookmark) to act as a negative control.

Images are written 8-bit windowed so they are viewable in a file browser; the
study/slice stays in the filename so the app still loads real 3D context.
"""
import csv
import json
import os
import os.path as osp
import collections

import cv2
import numpy as np

IMG_ROOT = 'data/DeepLesion/Images_png/Images_png'
DL_INFO = 'data/DeepLesion/DL_info.csv'
OUT = 'sample_images'
WINDOW = (-1024, 1050)
PER_REGION = 3
MIN_GAP = 12          # slices away from any bookmark for a negative control

TYPES = {1: 'bone', 2: 'abdomen', 3: 'mediastinum', 4: 'liver',
         5: 'lung', 6: 'kidney', 7: 'soft-tissue', 8: 'pelvis'}


def window(im):
    im = im.astype(np.float32) - 32768.0
    lo, hi = WINDOW
    return (np.clip((im - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def main():
    rows = list(csv.DictReader(open(DL_INFO)))
    have = set(os.listdir(IMG_ROOT))

    # every bookmarked slice per study, so negatives avoid all of them
    marked = collections.defaultdict(set)
    for r in rows:
        parts = r['File_name'].split('_')
        marked['_'.join(parts[:3])].add(int(parts[3][:-4]))

    by_type = collections.defaultdict(list)
    for r in rows:
        t = int(r['Coarse_lesion_type'])
        if t not in TYPES:
            continue
        parts = r['File_name'].split('_')
        study, sl = '_'.join(parts[:3]), parts[3]
        if study not in have or not osp.exists(osp.join(IMG_ROOT, study, sl)):
            continue
        long_px, short_px = [float(x) for x in r['Lesion_diameters_Pixel_'].split(',')]
        sx = float(r['Spacing_mm_px_'].split(',')[0])
        by_type[t].append(dict(
            study=study, slice=sl, rel='%s/%s' % (study, sl),
            long_mm=round(long_px * sx, 1), short_mm=round(short_px * sx, 1),
            split=int(r['Train_Val_Test']),
            noisy=int(r['Possibly_noisy']),
            bbox=[round(float(v), 1) for v in r['Bounding_boxes'].split(',')],
            spacing=sx,
        ))

    os.makedirs(OUT, exist_ok=True)
    for old in os.listdir(OUT):
        if old.endswith('.png'):
            os.remove(osp.join(OUT, old))

    manifest = []
    for t, region in TYPES.items():
        cands = by_type.get(t, [])
        # Spread the picks across the size range instead of taking the biggest:
        # one sub-2cm, one mid, one bulky, so the size bands actually differ.
        buckets = [(0, 20), (20, 40), (40, 1e9)]
        seen_studies = set()
        picked = []
        for lo, hi in buckets:
            pool = [c for c in cands
                    if lo <= c['long_mm'] < hi and c['study'] not in seen_studies]
            # unseen test split first, then clean annotations, then mid-bucket size
            pool.sort(key=lambda c: (c['split'] != 3, c['noisy'],
                                     abs(c['long_mm'] - (lo + min(hi, 60)) / 2)))
            if pool:
                seen_studies.add(pool[0]['study'])
                picked.append(pool[0])
        # top up from anywhere if a bucket was empty for this region
        for c in sorted(cands, key=lambda c: (c['split'] != 3, c['noisy'])):
            if len(picked) >= PER_REGION:
                break
            if c['study'] not in seen_studies:
                seen_studies.add(c['study'])
                picked.append(c)
        picked = picked[:PER_REGION]
        picked.sort(key=lambda c: c['long_mm'])

        for c in picked:
            name = '%s__%s__%s.png' % (region, c['study'], c['slice'][:-4])
            raw = cv2.imread(osp.join(IMG_ROOT, c['rel']), -1)
            cv2.imwrite(osp.join(OUT, name), window(raw))
            manifest.append(dict(
                file=name, region=region, source=c['rel'], kind='lesion',
                long_mm=c['long_mm'], short_mm=c['short_mm'],
                spacing=c['spacing'], split=c['split'], noisy=c['noisy']))

        # one lesion-free slice from the same anatomy; search every study that
        # contributed a lesion here, not just the first, and keep the widest gap
        best = None
        for c in cands:
            folder = osp.join(IMG_ROOT, c['study'])
            if not osp.isdir(folder):
                continue
            slices = sorted(int(f[:-4]) for f in os.listdir(folder)
                            if f.endswith('.png'))
            if len(slices) < 12:
                continue
            # fewer bookmarks in the study -> less chance of unmarked disease
            penalty = len(marked[c['study']])
            for s in slices:
                if not (slices[0] + 3 < s < slices[-1] - 3):
                    continue
                gap = min(abs(s - m) for m in marked[c['study']])
                score = (gap, -penalty)
                if best is None or score > best[0]:
                    best = (score, c['study'], s, c['spacing'])
        if best:
            best = (best[0][0], best[1], best[2], best[3])
        if best and best[0] >= MIN_GAP:
            gap, study, s, sp = best
            name = '%s__normal__%s__%03d.png' % (region, study, s)
            raw = cv2.imread(osp.join(IMG_ROOT, study, '%03d.png' % s), -1)
            cv2.imwrite(osp.join(OUT, name), window(raw))
            manifest.append(dict(
                file=name, region=region,
                source='%s/%03d.png' % (study, s), kind='normal',
                gap_from_bookmark=gap, spacing=sp))
        else:
            print('  ! no clean negative for %s (best gap %s)'
                  % (region, best[0] if best else 'n/a'))

    json.dump(manifest, open(osp.join(OUT, 'regions.json'), 'w'), indent=1)
    per = collections.Counter((m['region'], m['kind']) for m in manifest)
    print('%-13s %7s %7s' % ('region', 'lesion', 'normal'))
    for region in TYPES.values():
        print('%-13s %7d %7d' % (region, per[(region, 'lesion')],
                                 per[(region, 'normal')]))
    print('\n%d images -> %s/' % (len(manifest), OUT))


if __name__ == '__main__':
    main()
