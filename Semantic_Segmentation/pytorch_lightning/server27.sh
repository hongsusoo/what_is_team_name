#python train.py --gpus 2 --batch_size 16 --num_workers 8
#python train.py --gpus 2 --batch_size 32 --num_workers 8
#python train.py --gpus 2 --batch_size 16 --num_workers 8 --fp16 True
#python train.py --gpus 2 --batch_size 32 --num_workers 8 --fp16 True

python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --epochs 50 --learning_rate 0.0000001 --train_json_path "../data/train0_all.json" --val_json_path "../data/val0_all.json"
python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --epochs 50 --learning_rate 0.0000001 --train_json_path "../data/train1_all.json" --val_json_path "../data/val1_all.json"
python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --epochs 50 --learning_rate 0.0000001 --train_json_path "../data/train2_all.json" --val_json_path "../data/val2_all.json"
python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --epochs 50 --learning_rate 0.0000001 --train_json_path "../data/train3_all.json" --val_json_path "../data/val3_all.json"
python train.py --gpus 2 --archi DeepLabV3Plus --backbone se_resnext101_32x4d --epochs 50 --learning_rate 0.0000001 --train_json_path "../data/train4_all.json" --val_json_path "../data/val4_all.json"
