#python train.py --project recycle_trash_OD --name yolov5s --img 1024 --batch 16  --save-period 10 --multi-scale --noval --data data/recycle.yaml --weights yolov5s.pt --device 0
#python train.py --project recycle_trash_OD --name yolov5m --img 1024 --batch 16  --save-period 10 --multi-scale --noval --data data/recycle.yaml --weights yolov5m.pt --device 0
python train.py --project recycle_trash_OD --name yolov5l --img 1024 --batch 8  --save-period 10 --multi-scale --noval --data data/recycle.yaml --weights yolov5l.pt --device 0
python train.py --project recycle_trash_OD --name yolov5x --img 1024 --batch 8  --save-period 10 --multi-scale --noval --data data/recycle.yaml --weights yolov5x.pt --device 0