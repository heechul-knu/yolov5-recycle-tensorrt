from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
cocoGt = COCO('./saving_nomalize.json')
cocoDt = cocoGt.loadRes('./result_ann.json')
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()