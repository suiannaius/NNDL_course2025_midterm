import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from mmengine import Config
from mmengine.runner import Runner

def train_model(config_path):
    cfg = Config.fromfile(config_path)
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

def main():
    # mask_rcnn_config_path = os.path.join('voc_detection', 'mask_rcnn_voc.py')
    # train_model(mask_rcnn_config_path)

    sparse_rcnn_config_path = os.path.join('voc_detection', 'sparse_rcnn_voc.py')
    train_model(sparse_rcnn_config_path)

if __name__ == '__main__':
    main()