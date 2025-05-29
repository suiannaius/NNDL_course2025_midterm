from torchvision import datasets
import os
import torchvision.datasets as datasets

def download_vocdataset(voc_root='./data'):
    os.makedirs(voc_root, exist_ok=True)
    datasets.VOCSegmentation(root=voc_root, year='2007', image_set='test', download=True)
    datasets.VOCSegmentation(root=voc_root, year='2007', image_set='trainval', download=True)
    datasets.VOCSegmentation(root=voc_root, year='2012', image_set='trainval', download=True)
    print("Completed, data in:", os.path.abspath(voc_root))

def prepare_project_structure():
    work_dir = './voc_detection'
    os.makedirs(work_dir, exist_ok=True)
    model_dirs = ['mask_rcnn', 'sparse_rcnn']
    for model_dir in model_dirs:
        os.makedirs(os.path.join(work_dir, model_dir), exist_ok=True)
    for model_dir in model_dirs:
        os.makedirs(os.path.join(work_dir, f'{model_dir}_visualizations'), exist_ok=True)

def main():
    download_vocdataset()
    prepare_project_structure()

if __name__ == '__main__':
    main()
    