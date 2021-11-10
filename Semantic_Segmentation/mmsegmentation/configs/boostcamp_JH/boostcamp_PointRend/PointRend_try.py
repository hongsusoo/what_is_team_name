
_base_ = [
    '../_base_/models/pointrend_r50.py', '../boostcamp/dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]


model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", init_kwargs=dict(project="Naver_Segmentation", name="pointrend_r101_512x1024_80k_cityscapes_test_JH")),
    ],
)