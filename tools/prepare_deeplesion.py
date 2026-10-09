"""Prepare a local DeepLesion subset for MP3D.

1. Extracts the downloaded ``Images_png_XX.zip`` archives into the layout the
   configs expect (``data/DeepLesion/Images_png/Images_png/<study>/<slice>.png``).
2. Filters the COCO-style annotation files down to the studies that are
   actually present on disk, so training/testing works on a partial download.
"""
import argparse
import json
import os
import os.path as osp
import zipfile


def extract(zip_dir, out_dir):
    zips = sorted(f for f in os.listdir(zip_dir) if f.endswith('.zip'))
    if not zips:
        print('no zip files found in', zip_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    for name in zips:
        path = osp.join(zip_dir, name)
        if not zipfile.is_zipfile(path):
            # still downloading, or a truncated transfer
            print('%s: incomplete archive, skipping' % name)
            continue
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
            # archives are packed as Images_png/<study>/<slice>.png
            todo = [m for m in members
                    if not osp.exists(osp.join(out_dir, m)) and not m.endswith('/')]
            print('%s: %d entries, %d to extract' % (name, len(members), len(todo)))
            for i, m in enumerate(todo):
                zf.extract(m, out_dir)
                if (i + 1) % 5000 == 0:
                    print('  %d/%d' % (i + 1, len(todo)))


def subset_annotations(ann_dir, img_root, splits):
    available = set(os.listdir(img_root)) if osp.isdir(img_root) else set()
    print('studies on disk:', len(available))
    for split in splits:
        src = osp.join(ann_dir, 'deeplesion_%s.json' % split)
        if not osp.exists(src):
            continue
        data = json.load(open(src))
        images = [im for im in data['images']
                  if im['file_name'].split('/')[0] in available
                  and osp.exists(osp.join(img_root, im['file_name']))]
        keep_ids = {im['id'] for im in images}
        anns = [a for a in data['annotations'] if a['image_id'] in keep_ids]
        out = dict(data)
        out['images'] = images
        out['annotations'] = anns
        dst = osp.join(ann_dir, 'deeplesion_%s_local.json' % split)
        json.dump(out, open(dst, 'w'))
        print('%-5s -> %5d/%5d images, %5d/%5d lesions  (%s)'
              % (split, len(images), len(data['images']),
                 len(anns), len(data['annotations']), osp.basename(dst)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/DeepLesion')
    p.add_argument('--skip-extract', action='store_true')
    args = p.parse_args()

    zip_dir = osp.join(args.root, 'zips')
    img_out = osp.join(args.root, 'Images_png')
    if not args.skip_extract:
        extract(zip_dir, img_out)
    subset_annotations(osp.join(args.root, 'annotation'),
                       osp.join(img_out, 'Images_png'),
                       ['train', 'val', 'test'])


if __name__ == '__main__':
    main()
