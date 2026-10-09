"""Check that this Python environment can run MP3D, and report the device.

Imports every package the web app needs and runs mmcv's compiled operators,
which is where a broken install (missing Visual C++ runtime, wrong wheel)
shows up. Also re-applies the one site-packages fix this project relies on
(see RUNNING.md, "Fixes needed to make the repo run"), so a freshly built venv
behaves like the original one.

Exit code 0 when everything works (GPU or CPU), 1 otherwise.
"""
import sys

MMCV_ENV_ORIGINAL = "    except subprocess.CalledProcessError:\n        env_info['GCC'] = 'n/a'"
MMCV_ENV_FIXED = ("    except (subprocess.CalledProcessError, Exception):\n"
                  "        # no C compiler installed - this is only used for the env report\n"
                  "        env_info['GCC'] = 'n/a'")


def patch_mmcv_env():
    """Stop mmcv's collect_env() aborting tools/train.py when MSVC is absent."""
    import mmcv.utils.env as env
    with open(env.__file__, encoding='utf-8') as f:
        src = f.read()
    if src.count(MMCV_ENV_ORIGINAL) == 1:
        with open(env.__file__, 'w', encoding='utf-8') as f:
            f.write(src.replace(MMCV_ENV_ORIGINAL, MMCV_ENV_FIXED))
        print('applied the mmcv collect_env fix')


def main():
    try:
        import cv2  # noqa: F401
        import flask  # noqa: F401
        import torch
        import mmcv
        from mmcv.ops import RoIAlign, nms
        import mmdet
    except Exception as exc:
        print('ERROR: the Python environment is incomplete (%s: %s)'
              % (type(exc).__name__, exc))
        return 1
    print('Python %s, torch %s, mmcv %s, mmdet %s'
          % (sys.version.split()[0], torch.__version__, mmcv.__version__,
             mmdet.__version__))

    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    rois = torch.tensor([[0., 2., 2., 20., 20.]])
    try:
        nms(boxes, scores, 0.5)
        RoIAlign((7, 7), 1.0, 0)(torch.rand(1, 4, 32, 32), rois)
    except Exception as exc:
        print('ERROR: mmcv operators do not run (%s: %s)' % (type(exc).__name__, exc))
        return 1
    patch_mmcv_env()

    if not torch.cuda.is_available():
        print('No usable NVIDIA GPU: the model will run on the CPU '
              '(a few seconds per image).')
        return 0
    name = torch.cuda.get_device_name(0)
    try:
        x = torch.rand(1, 4, 3, 32, 32, device='cuda')
        y = torch.nn.GroupNorm(4, 8).cuda()(torch.nn.Conv3d(4, 8, 3, padding=1).cuda()(x))
        nms(boxes.cuda(), scores.cuda(), 0.5)
        RoIAlign((7, 7), 1.0, 0)(y[:, :, 1], rois.cuda())
        torch.cuda.synchronize()
    except Exception as exc:
        print('NVIDIA GPU %s found, but this PyTorch build cannot use it (%s); '
              'the model will run on the CPU.' % (name, type(exc).__name__))
        return 0
    print('NVIDIA GPU: %s' % name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
