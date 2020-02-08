#-*- coding: utf-8 -*-
# visualization code for sgcls task
# in KERN root dir, run python visualization/visualize_sgdet.py -cache_dir caches/kern_sgdet.pkl -save_dir visualization/saves

from collections import defaultdict
import gc 
import os
import json

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

parser.add_argument(
    '-split',
    dest='split',
    help='specify the split',
    type=str,
    default='VIST_example'
)

args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
image_dir = os.path.join(args.save_dir, 'images')
graph_dir = os.path.join(args.save_dir, 'graphs')
GNN_input_dir = os.path.join(args.save_dir, 'GNN_input')
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(GNN_input_dir, exist_ok=True)
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

new2old_mapping_dict = {}


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
    imgID = fn.split('/')[-1].split('.')[0]
    # print (imgID)
    
    im = mpimg.imread(fn)
    if not im.shape:
        print ('skipping file bcs empty image: ', prediction_id)
        return '', '', '', '', ''
    
    max_len = max(im.shape)
    scale = BOX_SCALE / max_len
    
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
    obj_thres_filtered = set()
    for new_obj_id, order_idx in enumerate(new_order_idx):
        rois[new_obj_id] = old_rois[order_idx]
        pred_obj_scores[new_obj_id] = old_pred_obj_scores[order_idx]
        pred_classes[new_obj_id] = old_pred_classes[order_idx]
        old2new_mapping[order_idx] = new_obj_id
        if pred_obj_scores[new_obj_id] < obj_thres:
            obj_thres_filtered.add(new_obj_id)
    for rel_id, rel in enumerate(pred_rel_inds):
        pred_rel_inds[rel_id][0] = old2new_mapping[rel[0]]
        pred_rel_inds[rel_id][1] = old2new_mapping[rel[1]]
    
    new2old_mapping = dict(map(reversed, old2new_mapping.items()))
    #print (old2new_mapping)
    #print (new2old_mapping)
    
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
    
    # draw graph
    sg_save_fn = os.path.join(graph_dir, fn.split('/')[-1].split('.')[-2])
    #u = Digraph('sg', filename=sg_save_fn, format=save_format)
    #u.attr('node', shape='box')
    #u.body.append('size="12,12"')
    #u.body.append('rankdir="LR"')
    
    # init GNN input
    nodes_list = [imgID]
    labels_list = []
    node1_list = []
    node2_list = []
    node_label_list = ['background']
    
    # add backgroup
    #u.node('0', label=imgID, color='forestgreen')
    name_list = []
    name_list_pred = []
    flag_has_node = False
    for i, l in enumerate(labels):
        if i in rel_inds and i not in obj_thres_filtered:
            id_w_background = i + 1
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
            
            old_id = new2old_mapping[i]
            node = imgID + '_' + str(old_id)
            nodes_list.append(node)
            labels_list.append('near')
            # parent
            node1_list.append('0')
            # child
            node2_list.append(str(id_w_background))
            
            node_label = ind_to_classes[pred_classes[i]]
            node_label_list.append(node_label)
            
            # PROOFREAD of object labels
            # label_by_old_id = ind_to_classes[old_pred_classes[old_id]]
            # u.node(str(id_w_background), label=obj_name_pred+' ('+ node + ')'+' ('+ label_by_old_id + ')', color='forestgreen')
            #u.node(str(id_w_background), label=obj_name_pred+' ('+ node + ')', color='forestgreen')
            #u.edge('0', str(id_w_background), label='near', color='forestgreen')
            
            flag_has_node = True
    if not flag_has_node:
        return '', '', '', '', ''
    
    for pred_rel in pred_rels: 
        if pred_rel[0] in rel_inds and pred_rel[1] in rel_inds and pred_rel[0] not in obj_thres_filtered and pred_rel[1] not in obj_thres_filtered:
            id0_w_background = pred_rel[0] + 1
            id1_w_background = pred_rel[1] + 1
            edge_label = ind_to_predicates[pred_rel[2]]

            labels_list.append(edge_label)
            # parent
            node1_list.append(str(id0_w_background))
            # child
            node2_list.append(str(id1_w_background))
            
            #u.edge(str(id0_w_background), str(id1_w_background), label=edge_label, color='forestgreen')
    
    # u.render(view=False, cleanup=False)
    
    node_label_s = '\t'.join(node_label_list)
    nodes_s = '\t'.join(nodes_list)
    labels_s = '\t'.join(labels_list)
    node1_s = '\t'.join(node1_list)
    node2_s = '\t'.join(node2_list)
    
    new2old_mapping_dict[imgID] = new2old_mapping
    
    return node_label_s, nodes_s, labels_s, node1_s, node2_s

    
with open(args.cache_dir, 'rb') as f:
    all_pred_entries = pkl.load(f)
print ('Loaded!')
print ('Obj label 0 is: ', ind_to_predicates[0])

GCN_input_dict = {}
all_node_label_list = []
all_nodes_list = []
all_labels_list = []
all_node1_list = []
all_node2_list = []
for i, pred_entry in enumerate(tqdm(all_pred_entries)):
    filename = test.filenames[i]
    imgID = filename.split('/')[-1].split('.')[0]
    
    if pred_entry == {}:
        print ('skipping file bcs empty entry: ', i)
        continue
    node_label_s, nodes_s, labels_s, node1_s, node2_s = visualize_pred_gt(pred_entry, i, filename, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k_rel=50, obj_thres=0.2, iou_thres=0.5)
    all_node_label_list.append(node_label_s)
    all_nodes_list.append(nodes_s)
    all_labels_list.append(labels_s)
    all_node1_list.append(node1_s)
    all_node2_list.append(node2_s)
    
    GCN_input = {}
    GCN_input['node_label'] = node_label_s
    GCN_input['node'] = nodes_s
    GCN_input['edge_label'] = labels_s
    GCN_input['parent_idx'] = node1_s
    GCN_input['child_idx'] = node2_s
    GCN_input_dict[imgID] = GCN_input
    
    
# print (all_node_label_list)

input_fp = os.path.join(GNN_input_dir, args.split + '_inputs.txt')
nodes_fp = os.path.join(GNN_input_dir, args.split + '_nodes.txt')
labels_fp = os.path.join(GNN_input_dir, args.split + '_labels.txt')
node1s_fp = os.path.join(GNN_input_dir, args.split + '_node1s.txt')
node2s_fp = os.path.join(GNN_input_dir, args.split + '_node2s.txt')
new2old_fp = os.path.join(GNN_input_dir, args.split + '_new2old.pkl')
GCN_input_dict_fp = os.path.join(GNN_input_dir, args.split + '_GCN_input_dict.pkl')

print ('writing input data...')
input_file_stream = '\n'.join(all_node_label_list)
input_file_stream += '\n'
with open(input_fp, 'w',  encoding='utf8') as f:
    f.write(input_file_stream)      
        
# write GNN graph files
print ('writing output nodes...')
nodes_file_stream = '\n'.join(all_nodes_list)
nodes_file_stream += '\n'
with open(nodes_fp, 'w', encoding='utf8') as f:
    f.write(nodes_file_stream)

print ('writing output labels...')
labels_file_stream = '\n'.join(all_labels_list)
labels_file_stream += '\n'
with open(labels_fp, 'w') as f:
    f.write(labels_file_stream)

print ('writing output node1...')
node1s_file_stream = '\n'.join(all_node1_list)
node1s_file_stream += '\n'
with open(node1s_fp, 'w') as f:
    f.write(node1s_file_stream)

print ('writing output node2...')
node2s_file_stream = '\n'.join(all_node2_list)
node2s_file_stream += '\n'
with open(node2s_fp, 'w') as f:
    f.write(node2s_file_stream)

with open(new2old_fp, 'wb') as f:
    pkl.dump(new2old_mapping_dict, f)

with open(GCN_input_dict_fp, 'wb') as f:
    pkl.dump(GCN_input_dict, f)
    
    