model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='modified_P3D',
        depth=63,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='GN', requires_grad=True, num_groups=32),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mp3d63-d720bda1.pth',
            prefix='backbone.'),
        ST_struc=('A', 'B', 'C'),
        conversion_type='groupconv',
        conversion_slice_num=9,
        conv_cfg=dict(type='Conv3d')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mp3d63-d720bda1.pth',
            prefix='neck.')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', loss_weight=1.0, beta=0.1111111111111111),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/mp3d63-d720bda1.pth',
            prefix='rpn_head.')),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0, beta=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'LesionDataset'
data_root = 'data/DeepLesion/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFile_3DCE',
        to_float32=False,
        lesion_input=True,
        num_slice=9,
        zflip=True,
        window=[-1024, 1050]),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        multiscale_mode='value',
        img_scale=[(320, 320), (352, 352), (384, 384)],
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
        is_3d_input=True,
        num_slice=9),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle', is_3d_input=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile_3DCE',
        to_float32=False,
        lesion_input=True,
        num_slice=9,
        zflip=False,
        window=[-1024, 1050]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                is_3d_input=True,
                num_slice=9),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='LesionDataset',
        ann_file='data/DeepLesion/annotation/deeplesion_train_local.json',
        img_prefix='data/DeepLesion/Images_png/Images_png/',
        pipeline=[
            dict(
                type='LoadImageFromFile_3DCE',
                to_float32=False,
                lesion_input=True,
                num_slice=9,
                zflip=True,
                window=[-1024, 1050]),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                multiscale_mode='value',
                img_scale=[(320, 320), (352, 352), (384, 384)],
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                is_3d_input=True,
                num_slice=9),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle', is_3d_input=True),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='LesionDataset',
        ann_file='data/DeepLesion/annotation/deeplesion_val_local.json',
        img_prefix='data/DeepLesion/Images_png/Images_png/',
        pipeline=[
            dict(
                type='LoadImageFromFile_3DCE',
                to_float32=False,
                lesion_input=True,
                num_slice=9,
                zflip=False,
                window=[-1024, 1050]),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                        is_3d_input=True,
                        num_slice=9),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LesionDataset',
        ann_file='data/DeepLesion/annotation/deeplesion_test_local.json',
        img_prefix='data/DeepLesion/Images_png/Images_png/',
        pipeline=[
            dict(
                type='LoadImageFromFile_3DCE',
                to_float32=False,
                lesion_input=True,
                num_slice=9,
                zflip=False,
                window=[-1024, 1050]),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(384, 384),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                        is_3d_input=True,
                        num_slice=9),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img'], is_3d_input=True),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=4, metric='bbox')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.3333333333333333,
    step=[7, 10])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
fp16 = dict(loss_scale=dict(init_scale=512))
pretrained = 'checkpoints/mp3d63-d720bda1.pth'
NUM_SLICE = 9
TEST_SCALE = (384, 384)
img_prefix = 'data/DeepLesion/Images_png/Images_png/'
work_dir = 'work_dirs/mp3d_lesion'
auto_resume = False
gpu_ids = [0]
