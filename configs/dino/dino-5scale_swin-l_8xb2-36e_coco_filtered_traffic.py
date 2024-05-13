_base_ = './dino-5scale_swin-l_8xb2-36e_coco.py'

test_dataloader = dict(dataset=dict(
    ann_file='annotations/instances_val2017_filtered_traffic.json'
))
