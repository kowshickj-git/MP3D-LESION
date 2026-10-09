"""Download what the web app needs from the project's GitHub release.

    python tools/fetch_release_files.py

* the trained detector (work_dirs/mp3d_lesion_hires/epoch_7.pth)
* the DeepLesion annotations and DL_info.csv (ground-truth boxes, pixel spacing)
* the CT slices around each built-in sample, so the samples run with their
  real 9-slice 3D context (data/DeepLesion/Images_png/...)

Downloads resume after a dropped connection and every file is checked against
its SHA-256. Files already in place are skipped without touching the network,
so once this has succeeded the app runs offline.
"""
import hashlib
import os
import os.path as osp
import sys
import time
import urllib.error
import urllib.request
import zipfile

RELEASE = 'https://github.com/kowshickj-git/MP3D-LESION/releases/download/v1.0/'
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

# release asset, size, sha256, destination (a zip is extracted into its folder)
FILES = [
    ('mp3d_lesion_hires_epoch_7.pth', 360384145,
     '49030b2ca295bdbac88590bb7cd719e257464d2a923c9076a183e5143cd6b0d7',
     'work_dirs/mp3d_lesion_hires/epoch_7.pth'),
    ('DeepLesion_annotation.zip', 4444853,
     '8a6d1fe4731702dcca9f56810839d3d940e97941a40f3476587d11eff1b72409',
     'data/DeepLesion/'),
    ('DeepLesion_sample_slices.zip', 63174919,
     '05f48e54086425e5c7caec27b6e863c31cfcd84b6cc6abb02d6b47013767add9',
     'data/DeepLesion/'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(name, size, dest):
    """Fetch one asset to `dest`, resuming a partial `dest + '.part'`."""
    part = dest + '.part'
    for attempt in range(1, 21):
        have = osp.getsize(part) if osp.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        if have == size:
            break
        req = urllib.request.Request(RELEASE + name, headers={'User-Agent': 'mp3d-setup'})
        if have:
            req.add_header('Range', 'bytes=%d-' % have)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:   # server ignored the range: start over
                    have = 0
                with open(part, 'ab' if have else 'wb') as f:
                    shown = 0.0
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        f.write(block)
                        have += len(block)
                        if time.time() - shown > 1 or have == size:
                            shown = time.time()
                            print('\r  %-30s %6.1f / %.1f MB (%3d%%)'
                                  % (name, have / 1e6, size / 1e6, 100 * have // size),
                                  end='', flush=True)
            print()
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                sys.exit('ERROR: %s is not available on the release (HTTP %d).'
                         % (name, exc.code))
            wait = min(60, 5 * attempt)
            print('\n  server problem (HTTP %d), retrying in %d s' % (exc.code, wait), flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, OSError) as exc:
            wait = min(60, 5 * attempt)
            print('\n  connection problem (%s), retrying in %d s' % (exc, wait), flush=True)
            time.sleep(wait)
    else:
        sys.exit('ERROR: could not download %s. Check the internet connection '
                 'and run again; the download will resume.' % name)
    os.replace(part, dest)


def fetch(name, size, digest, target):
    is_zip = target.endswith('/')
    folder = osp.join(ROOT, target)
    dest = osp.join(folder, name) if is_zip else osp.join(ROOT, target)
    marker = osp.join(folder, '.%s.extracted' % name)
    if is_zip and osp.exists(marker) and open(marker).read().strip() == digest:
        print('  %-30s already in place' % name)
        return
    if not is_zip and osp.exists(dest) and osp.getsize(dest) == size:
        if sha256(dest) == digest:
            print('  %-30s already in place' % name)
            return
        os.remove(dest)

    os.makedirs(osp.dirname(dest), exist_ok=True)
    for attempt in (1, 2):
        download(name, size, dest)
        if sha256(dest) == digest:
            break
        os.remove(dest)
        print('  %s was corrupted in transit, downloading it again' % name)
    else:
        sys.exit('ERROR: %s keeps arriving corrupted; run again later.' % name)

    if is_zip:
        with zipfile.ZipFile(dest) as z:
            z.extractall(folder)
        os.remove(dest)
        with open(marker, 'w') as f:
            f.write(digest)
        print('  %-30s extracted into %s' % (name, target))
    else:
        print('  %-30s saved as %s' % (name, target))


def main():
    for name, size, digest, target in FILES:
        fetch(name, size, digest, target)


if __name__ == '__main__':
    main()
