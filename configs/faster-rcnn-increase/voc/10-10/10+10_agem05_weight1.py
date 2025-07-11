# model settings
_base_ = [
    '10+10.py'
]

model = dict(
    pseudo_label_setting=dict(
        is_use=True,
        alpha=1.,
    )
)

train_dataloader = dict(
    batch_size=16,
    num_workers=16
)

# training schedule for 1x
train_cfg = dict(type='GPR', max_epochs=5, val_interval=1, lamda=0.6, is_use=True)
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),
    type='OptimWrapper')