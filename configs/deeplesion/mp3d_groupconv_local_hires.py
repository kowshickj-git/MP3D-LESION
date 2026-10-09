"""Second-stage MP3D run: more data, closer to the paper's input resolution.

Run 1 (``mp3d_groupconv_local.py``, 320-384px, 1010 slices) reached
FROC@4FP 4.7% with mAP_large 11.3% but mAP_medium only 2.0% -- small lesions
were being lost to the downscaling. This config keeps the same 6GB budget by
dropping to 1 image/iter, and spends it on resolution instead, over the
doubled (parts 01-04) DeepLesion subset, warm-started from run 1.
"""
_base_ = ['./mp3d_groupconv_local.py']

# warm start from the 384px run instead of re-learning the head from scratch
load_from = 'work_dirs/mp3d_lesion/latest.pth'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
NUM_SLICE = 9

train_pipeline = [
    dict(type='LoadImageFromFile_3DCE', to_float32=False, lesion_input=True,
         num_slice=NUM_SLICE, zflip=True, window=[-1024, 1050]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', multiscale_mode='value',
         img_scale=[(448, 448), (480, 480), (512, 512)], keep_ratio=True),
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
    dict(type='MultiScaleFlipAug', img_scale=(512, 512), flip=False,
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

optimizer = dict(lr=0.00125)          # linear scaling for 1 image/iter
lr_config = dict(warmup_iters=300, step=[4, 6])
runner = dict(max_epochs=7)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=2, metric='bbox')
log_config = dict(interval=100)
