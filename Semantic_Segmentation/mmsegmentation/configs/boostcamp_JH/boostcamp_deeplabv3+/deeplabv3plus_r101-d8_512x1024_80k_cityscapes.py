# _base_ = './deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py'

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '/opt/ml/segmentation/mmsegmentation/configs/boostcamp/dataset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))


log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", init_kwargs=dict(project="Naver_Segmentation", name="deeplabv3plus_r101-d8_512x1024_80k_cityscapes_test_JH")),
    ],
)