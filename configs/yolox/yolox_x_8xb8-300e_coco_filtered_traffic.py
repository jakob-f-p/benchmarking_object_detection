_base_ = './yolox_x_8xb8-300e_coco.py'

test_dataloader = dict(dataset=dict(
    ann_file='annotations/instances_val2017_filtered_traffic.json'
))
