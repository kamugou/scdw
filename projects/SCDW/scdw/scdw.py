
import copy
import random
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from mmdet.models import  SemiBaseDetector
from mmengine import MessageHub
from mmengine.model import BaseDataPreprocessor





class FWL_CBRSampler:
    def __init__(self, labeled_meta: List[Dict], labeled_gt_labels: List[torch.Tensor],
                 alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.labeled_meta = labeled_meta
        self.labeled_gt_labels = labeled_gt_labels
        self.num_classes = self._infer_num_classes(labeled_gt_labels)
        self.N_total = self._compute_global_counts(labeled_gt_labels)
        self.wsi_counts = self._compute_wsi_counts()
        self.w_total = self._compute_w_total()
        self.wsi_weights = self._compute_wsi_weights()
        self.labeled_image_weights = self._compute_image_weights()

    def _infer_num_classes(self, labels):
        m = 0
        for t in labels:
            if t is None or len(t)==0: continue
            m = max(m, int(torch.max(t).item()))
        return m+1 if m>=0 else 1

    def _compute_global_counts(self, labels):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for t in labels:
            if t is None or len(t)==0: continue
            for v in t.cpu().numpy().astype(int):
                counts[v] += 1
        return counts

    def _compute_wsi_counts(self):
        wsi_counts = {}
        for meta, labels in zip(self.labeled_meta, self.labeled_gt_labels):
            wsi = meta.get('wsi_id', 'unknown')
            if wsi not in wsi_counts:
                wsi_counts[wsi] = np.zeros(self.num_classes, dtype=np.int64)
            if labels is None or len(labels)==0: continue
            for v in labels.cpu().numpy().astype(int):
                wsi_counts[wsi][v] += 1
        return wsi_counts

    def _compute_w_total(self):
        inv = 1.0 / (self.N_total.astype(float) + self.eps)
        return inv / (inv.sum() + 1e-12)

    def _compute_wsi_weights(self):
        res = {}
        for wsi, counts in self.wsi_counts.items():
            inv = 1.0 / (counts.astype(float) + self.eps)
            norm = inv / (inv.sum() + 1e-12)
            res[wsi] = norm
        return res

    def _compute_image_weights(self):
        weights = []
        for meta, labels in zip(self.labeled_meta, self.labeled_gt_labels):
            freq = np.zeros(self.num_classes, dtype=float)
            if labels is not None and len(labels)>0:
                for v in labels.cpu().numpy().astype(int):
                    freq[v] += 1
            total = freq.sum()
            f_im = freq / (total + 1e-9) if total>0 else np.ones(self.num_classes)/self.num_classes
            wsi = meta.get('wsi_id', 'unknown')
            wsi_w = self.wsi_weights.get(wsi, np.ones(self.num_classes)/self.num_classes)
            w_comb = self.alpha * wsi_w + self.beta * self.w_total
            w_m_final = float(np.dot(f_im, w_comb))
            weights.append(w_m_final)
        return np.array(weights, dtype=float)

    def sample_labeled(self, n:int):
        w = self.labeled_image_weights.copy()
        if w.sum() <= 0:
            probs = np.ones_like(w)/len(w)
        else:
            probs = w / w.sum()
        idx = list(np.random.choice(len(w), size=n, replace=len(w)<n, p=probs))
        return idx

    def update_with_pseudo(self, unlabeled_meta: List[Dict], pseudo_labels: List[Dict]):
        for meta, pseudo in zip(unlabeled_meta, pseudo_labels):
            wsi = meta.get('wsi_id', 'unknown')
            if wsi not in self.wsi_counts:
                self.wsi_counts[wsi] = np.zeros(self.num_classes, dtype=np.int64)
            labs = pseudo.get('labels', np.zeros((0,), dtype=int))
            for v in labs.astype(int):
                if 0<=v<self.num_classes:
                    self.wsi_counts[wsi][v] += 1
        self.wsi_weights = self._compute_wsi_weights()
        self.labeled_image_weights = self._compute_image_weights()

class ComplexityAwareAug:
    def __init__(self, num_classes:int=2):
        self.num_classes = num_classes

    @staticmethod
    def entropy(p):
        p = p.copy()
        p = p / (p.sum()+1e-9)
        return float(-np.sum(p * np.log(p + 1e-9)))

    def compute_intensities(self, probs: np.ndarray):
        ent = self.entropy(probs)
        C = ent / math.log(max(2, self.num_classes))
        S_w = 0.05 + (0.3 - 0.05) * (1.0 - C)
        S_s = 0.2 + (0.8 - 0.2) * (1.0 - C)
        return S_w, S_s

    def apply_weak(self, img: torch.Tensor, intensity: float):
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.5:
            factor = 1.0 + (random.random()*2-1)*intensity
            img = img * factor
        if random.random() < 0.3:
            img = img + torch.randn_like(img) * (0.01 * intensity)
        return img.clamp(0.0, 1.0)

    def apply_strong(self, img: torch.Tensor, intensity: float):
        if random.random() < 0.5:
            k = random.choice([1,2,3])
            img = torch.rot90(img, k, dims=(1,2))
        if random.random() < 0.6:
            img = img + torch.randn_like(img) * (0.02 * intensity)
        if random.random() < 0.5:
            img = img * (1.0 + (random.random()*2-1)*(intensity*1.5))
        return img.clamp(0.0, 1.0)

    def batch_transform(self, imgs: torch.Tensor, pseudo_labels: List[Dict], metas: List[Dict], mode='auto'):
        B = imgs.shape[0]
        out = []
        for i in range(B):
            pl = pseudo_labels[i]
            labs = pl.get('labels', np.zeros((0,), dtype=int))
            probs = np.ones(self.num_classes)/self.num_classes
            if labs.size > 0:
                counts = np.zeros(self.num_classes)
                for v in labs.astype(int):
                    counts[v]+=1
                probs = counts / (counts.sum()+1e-9)
            S_w, S_s = self.compute_intensities(probs)
            img = imgs[i]
            if mode=='weak':
                t = self.apply_weak(img, S_w)
            elif mode=='strong':
                t = self.apply_strong(img, S_s)
            else:
                C = self.entropy(probs)/math.log(max(2,self.num_classes))
                t = self.apply_weak(img, S_w) if C>0.5 else self.apply_strong(img, S_s)
            out.append(t)
        return torch.stack(out, dim=0)

class WATPL:
    def __init__(self, num_classes:int=2, t_min:float=0.3, t_max:float=0.85, pop_size:int=16, generations:int=8):
        self.num_classes=num_classes
        self.t_min=t_min
        self.t_max=t_max
        self.pop_size=pop_size
        self.generations=generations
        self.labeled_dist_by_wsilabel = {}

    def set_labeled_distribution(self, d:Dict):
        self.labeled_dist_by_wsilabel = {k: (v / (v.sum()+1e-9)) for k,v in d.items()}

    def _compute_q(self, preds_for_wsi, tau_vec):
        counts = np.zeros(self.num_classes, dtype=float)
        total = 0.0
        for boxes, labels, scores in preds_for_wsi:
            if boxes is None or boxes.size==0:
                continue
            for lbl, sc in zip(labels.astype(int), scores.astype(float)):
                if 0<=lbl<self.num_classes:
                    total += 1.0
                    if sc >= tau_vec[lbl]:
                        counts[lbl] += 1.0
        if total==0:
            return np.ones(self.num_classes)/self.num_classes
        q = counts/(total+1e-9)
        q = (q+1e-6) / (q.sum()+1e-6*self.num_classes)
        return q

    def _fitness(self, tau_vec, preds_for_wsi, P_target):
        q = self._compute_q(preds_for_wsi, tau_vec)
        P = (P_target + 1e-9) / (P_target.sum()+1e-9*self.num_classes)
        kl = np.sum(P * np.log((P+1e-9)/(q+1e-9)))
        return -float(kl)

    def ga_optimize(self, preds_for_wsi, P_target):
        C = self.num_classes
        pop = np.random.uniform(self.t_min, self.t_max, size=(self.pop_size, C))
        fitness = np.array([self._fitness(ind, preds_for_wsi, P_target) for ind in pop])
        for gen in range(self.generations):
            idx = np.argsort(-fitness)
            topk = max(2, self.pop_size//2)
            top = pop[idx[:topk]]
            children=[]
            while len(children) < (self.pop_size-topk):
                a = top[np.random.randint(0, topk)]
                b = top[np.random.randint(0, topk)]
                alpha = np.random.rand(C)
                child = alpha*a + (1-alpha)*b
                mask = (np.random.rand(C) < 0.15)
                child[mask] += np.random.normal(scale=0.03, size=mask.sum())
                child = np.clip(child, self.t_min, self.t_max)
                children.append(child)
            pop = np.vstack([top, np.stack(children, axis=0)])
            fitness = np.array([self._fitness(ind, preds_for_wsi, P_target) for ind in pop])
        best = pop[np.argmax(fitness)]
        return best

    def refine(self, teacher_outputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], unlabeled_img_metas: List[Dict]):
        groups={}
        for i, meta in enumerate(unlabeled_img_metas):
            wsi = meta.get('wsi_id','unknown')
            groups.setdefault(wsi, {'indices':[], 'preds':[], 'wsi_label': meta.get('wsi_label', None)})
            boxes, labels, scores = teacher_outputs[i]
            if boxes is None or boxes.size==0:
                boxes_arr = np.zeros((0,4))
                labels_arr = np.zeros((0,), dtype=int)
                scores_arr = np.zeros((0,), dtype=float)
            else:
                boxes_arr = np.array(boxes[:, :4], dtype=float)
                labels_arr = np.array(labels, dtype=int)
                scores_arr = np.array(scores, dtype=float)
            groups[wsi]['indices'].append(i)
            groups[wsi]['preds'].append((boxes_arr, labels_arr, scores_arr))

        refined = [None]*len(teacher_outputs)
        for wsi, info in groups.items():
            preds_for_wsi = info['preds']
            wsi_label = info.get('wsi_label', None)
            P_target = self.labeled_dist_by_wsilabel.get(wsi_label, np.ones(self.num_classes)/self.num_classes)
            if len(preds_for_wsi)==0:
                for idx in info['indices']:
                    refined[idx] = {'bboxes': np.zeros((0,4)), 'labels': np.zeros((0,), dtype=int), 'scores': np.zeros((0,))}
                continue
            try:
                best_tau = self.ga_optimize(preds_for_wsi, P_target)
            except Exception:
                best_tau = np.ones(self.num_classes)*0.5
            for idx, (boxes, labels, scores) in zip(info['indices'], preds_for_wsi):
                if boxes is None or boxes.size==0:
                    refined[idx] = {'bboxes':np.zeros((0,4)),'labels':np.zeros((0,),dtype=int),'scores':np.zeros((0,))}
                    continue
                keep_mask = np.array([scores[i] >= best_tau[labels[i]] for i in range(len(scores))], dtype=bool) if len(scores)>0 else np.array([],dtype=bool)
                if keep_mask.size>0:
                    kept_boxes = boxes[keep_mask]
                    kept_labels = labels[keep_mask]
                    kept_scores = scores[keep_mask]
                else:
                    kept_boxes = np.zeros((0,4))
                    kept_labels = np.zeros((0,), dtype=int)
                    kept_scores = np.zeros((0,))
                refined[idx] = {'bboxes': kept_boxes, 'labels': kept_labels, 'scores': kept_scores}
        for i in range(len(refined)):
            if refined[i] is None:
                refined[i] = {'bboxes': np.zeros((0,4)), 'labels': np.zeros((0,), dtype=int), 'scores': np.zeros((0,))}
        return refined

@MODELS.register_module()
class SCDW(SemiBaseDetector):

    def __init__(self,
                 detector: ConfigType,
                 data_preprocessor: Optional[dict] = None,
                 semi_train_cfg: Optional[dict] = None,
                 semi_test_cfg: Optional[dict] = None,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 ema_momentum: float = 0.999,
                 watpl_cfg: Optional[dict] = None):
        super().__init__()
        self.detector = detector
        self.teacher_detector = None
        self.ema_momentum = ema_momentum
        self.alpha = alpha
        self.beta = beta
        self.semi_train_cfg = semi_train_cfg or {}
        self.semi_test_cfg = semi_test_cfg or {}

        num_classes = getattr(self.detector.bbox_head, 'num_classes', None)
        self.ca_aug = ComplexityAwareAug(num_classes=num_classes)
        self.watpl = WATPL(num_classes=num_classes, **(watpl_cfg or {}))
        self.sampler = None

        try:
            if hasattr(self.detector, 'init_weights'):
                self.detector.init_weights()
        except Exception:
            pass


    def _init_teacher(self):
        if self.teacher_detector is None:
            self.teacher_detector = copy.deepcopy(self.detector)
            for p in self.teacher_detector.parameters():
                p.requires_grad = False
            self.teacher_detector.eval()

    def ema_update(self):
        if self.teacher_detector is None:
            self._init_teacher()
            return
        alpha = self.ema_momentum
        with torch.no_grad():
            s_params = dict(self.detector.named_parameters())
            t_params = dict(self.teacher_detector.named_parameters())
            for name, tp in t_params.items():
                sp = s_params.get(name, None)
                if sp is None:
                    continue
                tp.data.mul_(alpha).add_(sp.data, alpha=1.0-alpha)

    def _teacher_infer(self, imgs: torch.Tensor, img_metas: List[Dict]):
  
        self._init_teacher()
        outputs=[]
        with torch.no_grad():
            feats = self.teacher_detector.extract_feat(imgs)
            cls_scores, bbox_preds = self.teacher_detector.bbox_head(feats)
            dets = self.teacher_detector.bbox_head.get_bboxes(cls_scores, bbox_preds, img_metas, self.detector.test_cfg, rescale=False)
            for det in dets:
                if isinstance(det, tuple) and len(det)==2:
                    bboxes, labels = det
                    if bboxes is None or getattr(bboxes, 'size', 0)==0:
                        outputs.append((np.zeros((0,5)), np.zeros((0,), dtype=int), np.zeros((0,))))
                    else:
                        b = np.array(bboxes, dtype=float)
                        lbl = np.array(labels, dtype=int)
                        scores = b[:,4].astype(float)
                        outputs.append((b, lbl, scores))
                else:
                    outputs.append((np.zeros((0,5)), np.zeros((0,), dtype=int), np.zeros((0,))))
        return outputs

    def forward_train(self, data_batch: dict, optimizer=None, **kwargs):

        losses = {}

        if 'imgs' in data_batch and 'img_metas' in data_batch:
            sup_imgs = data_batch['imgs']
            sup_metas = data_batch['img_metas']
            sup_bboxes = data_batch.get('gt_bboxes', None)
            sup_labels = data_batch.get('gt_labels', None)
        elif 'sup' in data_batch:
            sup = data_batch['sup']
            sup_imgs = sup.get('inputs', None) or sup.get('imgs', None)
            sup_metas = sup.get('data_samples', None) or sup.get('img_metas', None)
            sup_bboxes = sup.get('gt_bboxes', None)
            sup_labels = sup.get('gt_labels', None)
        else:
            raise RuntimeError('Unrecognized data_batch format for supervised branch.')

        feats_sup = self.detector.extract_feat(sup_imgs)
        sup_losses = self.detector.bbox_head.forward_train(feats_sup, sup_metas, sup_bboxes, sup_labels, None)
        for k,v in sup_losses.items():
            losses['sup.'+k] = v

        if self.sampler is None:
            try:
                self.sampler = FWL_CBRSampler(sup_metas, sup_labels, alpha=self.alpha, beta=self.beta)
            except Exception:
                self.sampler = None

        u_imgs = None
        u_metas = None
        u_student_imgs = None
        u_student_metas = None

        if 'unsup_teacher' in data_batch or 'unsup_student' in data_batch or 'unsup' in data_batch:
            ut = data_batch.get('unsup_teacher') or data_batch.get('unsup')
            us = data_batch.get('unsup_student')
            if isinstance(ut, dict):
                u_imgs = ut.get('inputs') or ut.get('imgs')
                u_metas = ut.get('data_samples') or ut.get('img_metas')
            else:
                u_imgs = ut
                u_metas = None
            if isinstance(us, dict):
                u_student_imgs = us.get('inputs') or us.get('imgs')
                u_student_metas = us.get('data_samples') or us.get('img_metas')
            else:
                u_student_imgs = us
                u_student_metas = None

        elif 'unlabeled_batch' in data_batch:
            ub = data_batch['unlabeled_batch']
            flat=[]
            if isinstance(ub, list) and len(ub)>0 and isinstance(ub[0], list):
                for sub in ub:
                    flat.extend(sub)
            else:
                flat = ub
            if len(flat)>0:
                u_imgs = torch.stack([u['unlabeled_img'] for u in flat], dim=0).to(sup_imgs.device)
                u_metas = [u['unlabeled_img_metas'] for u in flat]
                u_student_imgs = u_imgs.clone()
                u_student_metas = u_metas.copy()

        if u_imgs is None or (u_metas is None and u_student_metas is None):
            self.ema_update()
            return losses

        teacher_outs = self._teacher_infer(u_imgs, u_metas)

        labeled_dist_by_wsilabel = {}
        try:
            for meta, labels in zip(sup_metas, sup_labels):
                wsi_label = meta.get('wsi_label', None)
                if wsi_label is None:
                    continue
                arr = labeled_dist_by_wsilabel.get(wsi_label, np.zeros(self.ca_aug.num_classes, dtype=float))
                if labels is None or len(labels)==0:
                    continue
                for v in labels.cpu().numpy().astype(int):
                    if 0<=v<self.ca_aug.num_classes:
                        arr[v]+=1
                labeled_dist_by_wsilabel[wsi_label] = arr
            for k in list(labeled_dist_by_wsilabel.keys()):
                arr = labeled_dist_by_wsilabel[k]
                if arr.sum()<=0:
                    labeled_dist_by_wsilabel[k] = np.ones(self.ca_aug.num_classes)/self.ca_aug.num_classes
                else:
                    labeled_dist_by_wsilabel[k] = arr/arr.sum()
            self.watpl.set_labeled_distribution(labeled_dist_by_wsilabel)
        except Exception:
            pass

        refined = self.watpl.refine(teacher_outs, u_metas)

        try:
            if self.sampler is not None:
                self.sampler.update_with_pseudo(u_metas, refined)
        except Exception:
            pass

        if u_student_imgs is None:
            u_student_imgs = u_imgs.clone()
            u_student_metas = u_metas
        batch_u_s = self.ca_aug.batch_transform(u_student_imgs, refined, u_student_metas, mode='auto')

        feats_u = self.detector.extract_feat(batch_u_s)
        device = next(self.detector.parameters()).device
        pseudo_bboxes = [torch.from_numpy(r['bboxes']).to(device).float() for r in refined]
        pseudo_labels = [torch.from_numpy(r['labels']).to(device).long() for r in refined]

        try:
            unsup_losses = self.detector.bbox_head.forward_train(feats_u, u_student_metas, pseudo_bboxes, pseudo_labels, None)
            for k,v in unsup_losses.items():
                unsup_w = float(self.semi_train_cfg.get('unsup_weight', self.semi_train_cfg.get('unsup_weight', 0.5)))
                losses['unsup.'+k] = v * unsup_w
        except Exception:
            losses['unsup.proxy'] = torch.tensor(0.0, device=device)

        self.ema_update()

        return losses
