#-*- coding: utf-8 -*-
# visualization code for sgcls task
# in KERN root dir, run python visualization/visualize_sgdet.py -cache_dir caches/kern_sgdet.pkl -save_dir visualization/saves

from collections import defaultdict
import gc 
import os

from graphviz import Digraph
import numpy as np
import torch
from tqdm import tqdm
from config import ModelConfig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill as pkl

# conf = ModelConfig()

from dataloaders.vist import VISTDataLoader, VIST
from config import VIST_IMAGES, VIST_IM_FN, VIST_SGG_FN, VIST_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
import argparse

parser = argparse.ArgumentParser(description='visualization for sgdet task')
parser.add_argument(
    '-save_dir',
    dest='save_dir',
    help='dir to save visualization files',
    type=str,
    default='save/visualization'
)

parser.add_argument(
    '-cache_dir',
    dest='cache_dir',
    help='dir to load cache predicted results',
    type=str,
    default='caches/kern_sgdet.pkl'
)

args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
image_dir = os.path.join(args.save_dir, 'images')
graph_dir = os.path.join(args.save_dir, 'graphs')
os.makedirs(image_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)
mode = 'sgdet' 

# train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
#                         use_proposals=conf.use_proposals,
#                         filter_non_overlap=conf.mode == 'sgdet')
train, val, test = VIST.splits(num_val_im=-1, filter_duplicate_rels=True,
                        use_proposals=False,
                        filter_non_overlap=True)
val = test
# train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
#                                             batch_size=conf.batch_size,
#                                             num_workers=conf.num_workers,
#                                             num_gpus=conf.num_gpus)
_, test_loader = VISTDataLoader.splits(test, test, mode='rel',
                                            batch_size=1,
                                            num_workers=1,
                                            num_gpus=1)
ind_to_predicates = train.ind_to_predicates
ind_to_classes = train.ind_to_classes


def bb_intersection_over_union(boxA, boxB): 
    # determine the (x, y)-coordinates of the intersection rectangle 
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1]) 
    xB = min(boxA[2], boxB[2]) 
    yB = min(boxA[3], boxB[3]) 

    # compute the area of intersection rectangle 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) 

    # compute the area of both the prediction and ground-truth 
    # rectangles 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) 
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) 

    # compute the intersection over union by taking the intersection 
    # area and dividing it by the sum of prediction + ground-truth 
    # areas - the interesection area 
    iou = interArea / float(boxAArea + boxBArea - interArea) 

    # return the intersection over union value 
    return iou


def visualize_pred_gt(pred_entry, prediction_id, filename, ind_to_classes, ind_to_predicates, image_dir, graph_dir, top_k_rel=50, save_format='png', obj_thres=0.2, iou_thres=0.5):
    fn = filename
    im = mpimg.imread(fn)
    if not im.shape:
        print ('skipping file bcs empty image: ', prediction_id)
        return
    
    max_len = max(im.shape)
    scale = BOX_SCALE / max_len
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    # predict group
    old_rois = pred_entry['pred_boxes']
    old_pred_obj_scores = pred_entry['obj_scores']
    old_pred_classes = pred_entry['pred_classes']
    old_pred_rel_inds = pred_entry['pred_rel_inds']
    old_rel_scores = pred_entry['rel_scores']
    
    # sort by obj_detect score
    rois = old_rois.copy()
    pred_obj_scores = old_pred_obj_scores.copy()
    pred_classes = old_pred_classes.copy()
    pred_rel_inds = old_pred_rel_inds.copy()
    rel_scores = old_rel_scores.copy()
    new_order_idx = pred_obj_scores.argsort()[::-1]
    old2new_mapping = {}
    for new_obj_id, order_idx in enumerate(new_order_idx):
        rois[new_obj_id] = old_rois[order_idx]
        pred_obj_scores[new_obj_id] = old_pred_obj_scores[order_idx]
        pred_classes[new_obj_id] = old_pred_classes[order_idx]
        old2new_mapping[order_idx] = new_obj_id
    for rel_id, rel in enumerate(pred_rel_inds):
        pred_rel_inds[rel_id][0] = old2new_mapping[rel[0]]
        pred_rel_inds[rel_id][1] = old2new_mapping[rel[1]]
    
    # gt groups
    rois = rois / scale
    labels = pred_classes
    
    pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
    pred_rels = pred_rels[:top_k_rel]
    
    # Filter out dupes!
    # old_size = gt_rels.shape[0]
    all_rel_sets = defaultdict(list)
    for (o0, o1, r) in pred_rels:
        all_rel_sets[(o0, o1)].append(r)
    pred_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
    pred_rels = np.array(pred_rels)
    rels = pred_rels
    if rels.size > 0:
        rels_ = np.array(rels)
        rel_inds = rels_[:,:2].ravel().tolist()
    else:
        rel_inds = []
    
    object_name_list = []
    obj_count = np.zeros(151, dtype=np.int32)
    object_name_list_pred = []
    obj_count_pred = np.zeros(151, dtype=np.int32)
    for i, bbox in enumerate(rois):
        skip_this = False
        if int(labels[i]) == 0:
            # print ('skip background!!!')
            continue
        if i not in rel_inds:
            # print ('skip box with no relation predicted!!!')
            continue
        if pred_obj_scores[i] < obj_thres:
            # print ('under obj_thres!!!')
            continue
        for pre_bbox in rois[0:i]:
            if bb_intersection_over_union(bbox, pre_bbox) > iou_thres:
                # print ('over iou_thres!!!')
                skip_this = True
                break
        if skip_this:
            continue
        
        label_str = ind_to_classes[int(labels[i])]
        while label_str in object_name_list:
            obj_count[int(labels[i])] += 1
            label_str = label_str + '_' + str(obj_count[int(labels[i])])
        object_name_list.append(label_str)
        pred_classes_str = ind_to_classes[int(pred_classes[i])]
        while pred_classes_str in object_name_list_pred:
            obj_count_pred[int(pred_classes[i])] += 1
            pred_classes_str = pred_classes_str + '_' + str(obj_count_pred[int(pred_classes[i])])
        object_name_list_pred.append(pred_classes_str)
        if labels[i] == pred_classes[i]:
            ax.text(bbox[0], bbox[1] - 2,
                    label_str,
                    bbox=dict(facecolor='green', alpha=0.5),
                    fontsize=32, color='white')
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='green', linewidth=3.5)
            )
        else:
            ax.text(bbox[0], bbox[1] - 2,
                    pred_classes_str + ' ('+ label_str + ')',
                    bbox=dict(facecolor='red', alpha=0.5),
                    fontsize=32, color='white')   
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5)
                )
    
    # draw relations
    ax.axis('off')
    fig.tight_layout()

    image_save_fn = os.path.join(image_dir, fn.split('/')[-1].split('.')[-2]+'.'+save_format)
    plt.savefig(image_save_fn)
    plt.close()


    sg_save_fn = os.path.join(graph_dir, fn.split('/')[-1].split('.')[-2])
    u = Digraph('sg', filename=sg_save_fn, format=save_format)
    u.attr('node', shape='box')
    u.body.append('size="12,12"')
    u.body.append('rankdir="LR"')

    name_list = []
    name_list_pred = []
    flag_has_node = False

    for i, l in enumerate(labels):
        if i in rel_inds:
            name = ind_to_classes[l]
            name_pred = ind_to_classes[pred_classes[i]]
            name_suffix = 1
            name_suffix_pred = 1
            obj_name = name
            obj_name_pred = name_pred
            while obj_name in name_list:
                obj_name = name + '_' + str(name_suffix)
                name_suffix += 1
            while obj_name_pred in name_list_pred:
                obj_name_pred = name_pred + '_' + str(name_suffix_pred)
                name_suffix_pred += 1
            name_list.append(obj_name)
            name_list_pred.append(obj_name_pred)
            if l == pred_classes[i]:
                u.node(str(i), label=obj_name, color='forestgreen')
            else:
                u.node(str(i), label=obj_name_pred+' ('+ obj_name + ')', color='red')
            flag_has_node = True
    if not flag_has_node:
        return
    
#     pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
#     pred_rels = pred_rels[:top_k_rel]
    
    for pred_rel in pred_rels:
        for rel in rels:
            if pred_rel[0] == rel[0] and pred_rel[1] == rel[1]:
                u.edge(str(pred_rel[0]), str(pred_rel[1]), label=ind_to_predicates[pred_rel[2]], color='forestgreen')
                '''
                label_str = ind_to_predicates[pred_rel[2]] + ' (' + ind_to_predicates[rel[2]] + ')'
                u.edge(str(rel[0]), str(rel[1]), label=label_str, color='red')
                '''
                
    for rel in rels:
        flag_rel_has_pred = False
        for pred_rel in pred_rels:
            if pred_rel[0] == rel[0] and pred_rel[1] == rel[1]:
                flag_rel_has_pred = True
                break
        if not flag_rel_has_pred:
            u.edge(str(pred_rel[0]), str(pred_rel[1]), label=ind_to_predicates[pred_rel[2]], color='grey50')
    u.render(view=False, cleanup=True)
    
with open(args.cache_dir, 'rb') as f:
    all_pred_entries = pkl.load(f)
print ('Loaded!')
print ('Obj label 0 is: ', ind_to_predicates[0])

for i, pred_entry in enumerate(tqdm(all_pred_entries)):
    filename = test.filenames[i]
    
    # you could use these three lines of code to only visualize some images
    # if num_id == '2343586' or num_id == '2343599' or num_id == '2315539':
    #     visualize_pred_gt(pred_entry, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k_rel=50)
    
    if pred_entry == {}:
        print ('skipping file bcs empty entry: ', i)
        continue
    
    visualize_pred_gt(pred_entry, i, filename, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k_rel=50, obj_thres=0.2, iou_thres=0.5)
    
    
    