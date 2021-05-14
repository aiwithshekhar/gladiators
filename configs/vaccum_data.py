dataset_type = 'VaccumDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MinIoURandomCrop'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset_name='robot',
        data_split='all',
        split='train',
        split_ratio=0.85,
        seed=42,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset_name='robot',
        data_split='all',
        split='val',
        split_ratio=0.85,
        seed=42,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset_name='robot',
        data_split='all',
        split='test',
        split_ratio=0.85,
        seed=42,
        pipeline=test_pipeline)
)