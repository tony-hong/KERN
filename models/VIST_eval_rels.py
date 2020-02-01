import os

import numpy as np
import torch
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
import dill as pkl

from config import ModelConfig
from config import BOX_SCALE, IM_SCALE, RCNN_CHECKPOINT_FN
from lib.kern_model import KERN
from dataloaders.vist import VISTDataLoader, VIST

conf = ModelConfig()


train, val, test = VIST.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship

if conf.test:
    val = test
_, val_loader = VISTDataLoader.splits(val, test, mode='det',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                use_ggnn_rel=conf.use_ggnn_rel, ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
                ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim, ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
                use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge)


detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])


if conf.mode == 'sgdet':
    det_ckpt = torch.load(RCNN_CHECKPOINT_FN)['state_dict']
#     print (det_ckpt['bbox_fc.weight'].shape)
#     print (det_ckpt['bbox_fc.bias'].shape)
#     print (det_ckpt['score_fc.weight'].shape)
#     print (det_ckpt['score_fc.bias'].shape)
    detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
    detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
    detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
    detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])
    print ('Weight of object detector loaded! ')
    

all_pred_entries = []

def val_batch(batch_num, b): 
    det_res = detector[b]
    if not det_res:
        all_pred_entries.append({})
        return
    
    if conf.num_gpus == 1:
        det_res = [det_res]
    
    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)
    
detector.eval()
for val_b, batch in enumerate(tqdm(val_loader)):
    val_batch(conf.num_gpus*val_b, batch)

if conf.cache is not None:
    with open(conf.cache,'wb') as f:
        pkl.dump(all_pred_entries, f)

