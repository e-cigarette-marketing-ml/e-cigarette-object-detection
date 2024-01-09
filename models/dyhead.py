_base_ = '../external/mmdetection/configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco.py'

model = dict(bbox_head = dict(num_classes = 9))
dataset_type = 'CocoDataset'

data_root = '../data/annotations/'
classes = ('box', 'e-cigarette brand name', 'e-juice', 'e-juice flavor', 'mod', 'pod', 'smoke cloud', 'synthetic nicotine label', 'warning label nicotine')

img_prefix = '../data/images/'

# Dataset files created by notebooks/v7-annotations-to-coco.ipynb
data = dict(
    samples_per_gpu=1, # 2 couldn't fit into 48GB of VRAM. (each sample requires ~29GB)
    train = dict(type = 'RepeatDataset',
                 times = 1,
                 dataset =
                     dict(type = dataset_type,
                          img_prefix = img_prefix,
                          classes = classes,
                          ann_file = data_root + 'v7-coco-train.json')),
    val = dict(type=dataset_type,
               img_prefix = img_prefix, classes = classes,
               ann_file = data_root + 'v7-coco-val.json'),
    test = dict(type=dataset_type,
                img_prefix = img_prefix,
                classes = classes,
                ann_file = data_root + 'v7-coco-test.json'))

# Expected minibatch size is 16 images (4 per gpu * 4 gpus).
# We are only doing 2 per gpu * 2 gpus = 4.
# So the lr should be 1/4 of the expected LR of lr=0.01.
#optimizer = dict(lr = 0.0025)

log_config = dict(interval=50, hooks=[dict(type='TensorboardLoggerHook')])

# Checkpoint downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/
load_from = '../models/checkpoints/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'
