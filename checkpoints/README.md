# checkpoints

`mp3d63-d720bda1.pth` (87 MB) goes here: the COCO-supervised MP3D63 model from
the paper, used to warm-start training (RUNNING.md, section 3). It is too large
for the repository and is attached to the
[v1.0 release](https://github.com/kowshickj-git/MP3D-LESION/releases/tag/v1.0):

```bash
curl -L -o checkpoints/mp3d63-d720bda1.pth \
  https://github.com/kowshickj-git/MP3D-LESION/releases/download/v1.0/mp3d63-d720bda1.pth
```

Only training needs it; the web app does not.
