import torch
import torch.nn as nn
from scipy import optimize
import numpy as np
import torch.nn.functional as F
import json

class Align23dLoss:
    def __init__(self):
        self.get_similarity = nn.CosineSimilarity(dim = 1)
        self.num_classes = 11
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def get_align_23d_loss(self, aligned_feat, dataset, valid_img_idx, valid_img_weight, valid_img_feat, valid_word_idx, valid_word_feat, confidence_weight=True, voxel_pixel_align=True, voxel_word_align=True):
        # TODO can be optimized by scatter
        if dataset == "NYU":
            # aligned_feat = aligned_feat.flatten(2).squeeze().permute(1,0)
            aligned_feat = aligned_feat.flatten(2).permute(0,2,1)

        if voxel_pixel_align:
            ###### for full feat
            img_align_loss_batch = []
            for i in range(len(aligned_feat)):
                if dataset == "NYU":
                    aligned_feat_img = aligned_feat[i].index_select(0, valid_img_idx[i])
                ###### for selected feat
                elif dataset == "kitti":
                    aligned_feat_img = aligned_feat

                similarity = self.get_similarity(aligned_feat_img, valid_img_feat[i])
                img_align_loss = 1 - similarity
                ###### ablation study: w/ or w/o confidence
                if confidence_weight:
                    img_align_loss *= valid_img_weight[i]
                img_align_loss_batch.append(img_align_loss.mean())
            img_align_loss = torch.stack(img_align_loss_batch)

        else:
            img_align_loss = torch.tensor([0.])
        
        if voxel_word_align and dataset != "kitti":
            word_align_loss_batch = []
            for i in range(len(aligned_feat)):
                if aligned_feat.shape[1] != len(valid_word_idx[0]):
                    aligned_feat_word = aligned_feat[i].index_select(0, valid_word_idx[i])
                similarity = self.get_similarity(aligned_feat_word, valid_word_feat[i].squeeze())
                word_align_loss = 1 - similarity
                word_align_loss_batch.append(word_align_loss.mean())
            word_align_loss = torch.stack(word_align_loss_batch)
        else:
            word_align_loss = torch.tensor([0.])
        return img_align_loss.mean(), word_align_loss.mean()