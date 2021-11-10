import argparse

import albumentations as A
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description='Inference and make the csv file for submission to AI-Stage')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('output', help='output file path to save results')
    parser.add_argument('--test_json_path',
                        default='/opt/ml/segmentation/input/data/test.json',
                        help='path of test.json')
    parser.add_argument('--for_pseudo',
                        action='store_true',
                        help='If true, a csv file includes 512 x 512 images.')
    parser.add_argument('--out_raw_pkl_path',
                        type=str,  # actually, Optional[str]
                        defalut=None,
                        help='file_path of raw result of model output')

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.gpu_ids = [0]
    cfg.data.test.test_mode = True

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        drop_last=False,
        dist=False,
        shuffle=False)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader)

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(args.test_json_path)

    resize_transform = A.Compose([A.Resize(256, 256, 0)])  # cv2.INTER_NEAREST

    for i, out in enumerate(output):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        file_names.append(image_info['file_name'])

        if args.for_pseudo:
            out_fit_array = out
        else:
            out_fit_array = resize_transform(image=out)['image']
        prediction_string = ' '.join([str(pixel_pred) for pixel_pred in out_fit_array.flatten().tolist()])
        prediction_strings.append(prediction_string)

    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
    submission.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
