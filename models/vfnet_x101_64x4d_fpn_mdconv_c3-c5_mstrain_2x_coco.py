_base_ = '../external/mmdetection/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py'

model = dict(bbox_head = dict(num_classes = 9))

data_root = '../data/annotations/'
classes = ('box', 'e-cigarette brand name', 'e-juice', 'e-juice flavor', 'mod', 'pod', 'smoke cloud', 'synthetic nicotine label', 'warning label nicotine')

img_prefix = '../data/images/'

# Dataset files created by notebooks/v7-annotations-to-coco.ipynb
data = dict(
    train = dict(img_prefix = img_prefix, classes = classes,
        ann_file = data_root + 'v7-coco-train.json'),
    val = dict(img_prefix = img_prefix, classes = classes,
        ann_file = data_root + 'v7-coco-val.json'),
    test = dict(img_prefix = img_prefix, classes = classes,
        ann_file = data_root + 'v7-coco-test.json'))

# Expected minibatch size is 16 images (4 per gpu * 4 gpus).
# We are only doing 2 per gpu * 2 gpus = 4.
# So the lr should be 1/4 of the expected LR of lr=0.01.
optimizer = dict(lr = 0.0025)

log_config = dict(interval=50, hooks=[dict(type='TensorboardLoggerHook')])

# Checkpoint downloaded from https://github.com/open-mmlab/mmdetection/tree/master/configs/pvt/
load_from = '../models/checkpoints/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth'
