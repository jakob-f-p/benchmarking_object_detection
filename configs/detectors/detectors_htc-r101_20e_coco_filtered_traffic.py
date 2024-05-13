_base_ = './detectors_htc-r101_20e_coco.py'

test_dataloader = dict(dataset=dict(
    ann_file='annotations/instances_val2017_filtered_traffic.json'
))
