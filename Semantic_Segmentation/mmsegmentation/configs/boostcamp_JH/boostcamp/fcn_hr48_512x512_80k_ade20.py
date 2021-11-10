_base_ = [
    'schedule_160k.py',
    'dataset.py',
    'default_runtime.py',
    'fcn_hr18.py'
]

# _base_ = './fcn_hr18.py'

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", init_kwargs=dict(project="Naver_Segmentation", name="fcn_hr48_512x512_80k_ade20_test_JH")),
    ],
)