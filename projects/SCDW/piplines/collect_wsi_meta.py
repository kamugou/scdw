# projects/scdw/pipelines/collect_wsi_meta.py
from mmdet.registry import PIPELINES

@PIPELINES.register_module()
class CollectWSIMeta:
    """Collect WSI metadata from img_info into img_metas for model use."""
    def __call__(self, results):
        img_info = results.get('img_info', {})
        img_metas = results.get('img_metas', {})
        results.setdefault('img_metas', {})
        results['img_metas']['wsi_id'] = img_info.get('wsi_id', None)
        results['img_metas']['wsi_label'] = img_info.get('wsi_label', None)
        return results
