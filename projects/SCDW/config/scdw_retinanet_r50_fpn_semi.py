_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'mmdet::_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.scdw.scdw',
        'projects.scdw.datasets.collect_wsi_meta',
        'projects.scdw.datasets.semi_coco_dataset',
    ],
    allow_failed_imports=False
)
fwl_cbr_sampler = dict(
    type='FWLClassBalancedSampler',
    alpha=0.5, 
    beta=0.5,
    wsi_label_mapping={
        'ascus': 0, 
        'lsil': 1,
        'hsil': 2,
        'asch': 3
    }
)
detector = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256,512,1024,2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5,1.0,2.0],
            strides=[8,16,32,64,128]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[.0,.0,.0,.0], target_stds=[1.0,1.0,1.0,1.0]),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0, ignore_iof_thr=-1),
        allowed_border=-1, pos_weight=-1, debug=False
    ),
    test_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), min_bbox_size=0, score_thr=0.05, max_per_img=100)
)

model = dict(
    type='SCDW',
    detector=dict(
        type='RetinaNet',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        bbox_head=dict(
            type='RetinaHead',
            num_classes=4,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    semi_train_cfg=dict(
        watpl_cfg=dict(
            use_genetic_algorithm=True,
            num_generations=50,
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8
        ),
        caas_cfg=dict(
            min_weak_intensity=0.1,
            max_weak_intensity=0.3,
            min_strong_intensity=0.5, 
            max_strong_intensity=0.9
        ),
        cache_size=2000,
        mosaic=True,
        mosaic_weight=0.5,
        mosaic_shape=[(640, 640), (800, 800)],
        erase_patches=(1, 3),
        erase_ratio=(0.02, 0.2),
        erase_thr=0.5
    ),
    semi_test_cfg=dict(
        pseudo_label_score_thr=0.5,
        nms_thr=0.5
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

scale = [(1333, 400), (1333, 1200)]
color_space = 'imagenet'
geometric = 'imagenet'

sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='CollectWSIMeta'), 
    dict(type='PackDetInputs')
]

weak_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='CollectWSIMeta'),
    dict(type='PackDetInputs'),
]

strong_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomOrder', transforms=[
        dict(type='RandAugment', aug_space=color_space, aug_num=1),
        dict(type='RandAugment', aug_space=geometric, aug_num=1),
    ]),
    dict(type='RandomErasing', n_patches=(1,5), ratio=(0,0.2)),
    dict(type='CollectWSIMeta'),
    dict(type='PackDetInputs'),
]

unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadEmptyAnnotations'),
    dict(type='MultiBranch', unsup_teacher=weak_pipeline, unsup_student=strong_pipeline)
]

data_root = 'data/cervical/'

labeled_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    ann_file='annotations/instances_train_labeled.json',
    data_prefix=dict(img='train_labeled/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=sup_pipeline
)

unlabeled_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    ann_file='annotations/instances_unlabeled.json',
    data_prefix=dict(img='train_unlabeled/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline
)

train_dataloader = dict(
    batch_size=5, 
    num_workers=5,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=2,
        source_ratio=[1, 4] 
    ),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset])
)

val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.9995, interval=1, warm_up=0),
]

val_cfg = dict(type='TeacherStudentValLoop')

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
]
