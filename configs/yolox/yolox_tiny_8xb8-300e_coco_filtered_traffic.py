_base_ = './yolox_tiny_8xb8-300e_coco.py'

val_dataloader = dict(dataset=dict(
    ann_file='annotations/instances_val2017_filtered_traffic.json'
))

test_dataloader = dict(dataset=dict(
    ann_file='annotations/instances_val2017_filtered_traffic.json'
))
val_evaluator = [    dict(
        type='CocoMetric',
        ann_file='data/coco/annotations/instances_val2017.json',
        metric='bbox',
        metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'AR@100', 'AR@300',
                      'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000']
    )
]
test_evaluator = val_evaluator
