_base_ = [
    '../ddod/models/atss_iou_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu=4)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

fp16 = dict(loss_scale=512.)
