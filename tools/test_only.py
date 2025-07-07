import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import yaml
import argparse
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.model_helper import build_model
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import DataLoader
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger, set_random_seed

parser = argparse.ArgumentParser(description='MonoDETR Evaluation Only Script')
parser.add_argument('--config', dest='config', required=True, help='Path to yaml config file')
args = parser.parse_args()

def main():
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)

    set_random_seed(cfg.get('random_seed', 444))
    model_name = cfg['model_name']

    # 로그 파일 저장용 디렉토리
    log_output_path = os.path.join(cfg["trainer"]['save_path'], model_name)
    os.makedirs(log_output_path, exist_ok=True)

    log_file = os.path.join(log_output_path, 'eval.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    logger.info('############### Evaluation Only ###############')
    logger.info(f"Loading config from: {args.config}")

    # ✅ Test Loader 생성
    test_set = KITTI_Dataset(split=cfg['dataset']['test_split'], cfg=cfg['dataset'])
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 생성
    model, _ = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    # Tester 생성
    tester = Tester(
        cfg=cfg['tester'],
        model=model,
        dataloader=test_loader,
        logger=logger,
        train_cfg=cfg['trainer'],
        model_name=model_name
    )

    # checkpoint 경로 확인
    ckpt = cfg['tester']['checkpoint']
    if isinstance(ckpt, int):
        checkpoint_path = os.path.join(cfg["trainer"]['save_path'], model_name, f"checkpoint_epoch_{ckpt}.pth")
    else:
        checkpoint_path = ckpt

    logger.debug(f"[DEBUG] Using checkpoint: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"❌ Checkpoint not found: {checkpoint_path}"

    # Run test
    tester.test(checkpoint_path)

if __name__ == '__main__':
    main()
