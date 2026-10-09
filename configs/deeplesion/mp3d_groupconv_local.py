"""MP3D-63 group-conv config adapted for a single 6GB GPU + partial DeepLesion.

Differences from ``mp3d_groupconv.py`` (8x V100, batch 16):
  * pretrained checkpoint read from ``checkpoints/`` with the ``backbone.``
    prefix the released COCO model actually uses (it is an mmdet-1.x
    state_dict: backbone.* / neck.* / rpn_head.* / bbox_head.*), and the FPN
    and RPN head are warm-started from that same checkpoint;
  * smaller input scale + frozen stem/layer1 so activations fit in 6GB of
    VRAM (at 512-576px the run spills into host memory and crawls);
  * batch size 1, lr scaled accordingly, shorter schedule, and the
    ``*_local.json`` annotation subsets from ``tools/prepare_deeplesion.py``.
"""
_base_ = ['./mp3d_groupconv.py']

pretrained = 'checkpoints/mp3d63-d720bda1.pth'

model = dict(
    backbone=dict(
        frozen_stages=1,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='backbone.')),
    neck=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='neck.')),
    rpn_head=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained, prefix='rpn_head.')))

dataset_type = 'LesionDataset'
data_root = 'data/DeepLesion/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
NUM_SLICE = 9
TEST_SCALE = (384, 384)

train_pipeline = [
    dict(type='LoadImageFromFile_3DCE', to_float32=False, lesion_input=True,
         num_slice=NUM_SLICE, zflip=True, window=[-1024, 1050]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', multiscale_mode='value',
         img_scale=[(320, 320), (352, 352), (384, 384)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg, is_3d_input=True,
         num_slice=NUM_SLICE),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_3DCE', to_float32=False, lesion_input=True,
         num_slice=NUM_SLICE, zflip=False, window=[-1024, 1050]),
    dict(type='MultiScaleFlipAug', img_scale=TEST_SCALE, flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg, is_3d_input=True,
                  num_slice=NUM_SLICE),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
             dict(type='Collect', keys=['img']),
         ])
]

img_prefix = data_root + 'Images_png/Images_png/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(type=dataset_type, img_prefix=img_prefix,
               ann_file=data_root + 'annotation/deeplesion_train_local.json',
               pipeline=train_pipeline),
    val=dict(type=dataset_type, img_prefix=img_prefix,
             ann_file=data_root + 'annotation/deeplesion_val_local.json',
             pipeline=test_pipeline),
    test=dict(type=dataset_type, img_prefix=img_prefix,
              ann_file=data_root + 'annotation/deeplesion_test_local.json',
              pipeline=test_pipeline))

# lr scaled from the paper's 0.02 @ 8x2 down to 1 image/iter
optimizer = dict(lr=0.0025)
lr_config = dict(warmup_iters=300, step=[7, 10])
runner = dict(max_epochs=12)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=4, metric='bbox')
log_config = dict(interval=50)
