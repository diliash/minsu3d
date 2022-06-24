import torch
import functools
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.evaluation.instance_seg_helper import ScanNetEval, rle_encode
from lib.softgroup_ops.functions import softgroup_ops
from lib.loss import *
from model.common import ResidualBlock, VGGBlock, UBlock
from lib.evaluation.semantic_seg_helper import *


class SoftGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.DC = ScannetDatasetConfig(cfg)

        input_channel = cfg.model.use_coords * 3 + cfg.model.use_color * 3 + cfg.model.use_normal * 3
        m = cfg.model.m
        D = 3
        semantic_classes = cfg.data.classes
        self.instance_classes = semantic_classes - len(cfg.data.ignore_classes)
        blocks = cfg.model.blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual

        self.freeze_backbone = cfg.model.freeze_backbone
        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.grouping_radius = cfg.model.grouping_cfg.radius
        self.grouping_meanActive = cfg.model.grouping_cfg.mean_active
        self.grouping_npoint_threshold = cfg.model.grouping_cfg.npoint_thr

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock
        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        """
        Bottom-up Grouping Block
        """
        # 1. backbone U-Net
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(input_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m * c for c in blocks], sp_norm, block_reps, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        # 2.1 semantic prediction branch
        self.semantic_branch = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, semantic_classes)
        )

        # 2.2 offset prediction branch
        self.offset_branch = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, 3)
        )

        """
        Top-down Refinement Block
        """
        # 3 tiny U-Net
        self.tiny_unet = nn.Sequential(
            UBlock([m, 2 * m], sp_norm, 2, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        # 4.1 classification branch
        self.classification_branch = nn.Linear(m, self.instance_classes + 1)

        # 4.2 mask scoring branch
        self.mask_scoring_branch = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(inplace=True),
            nn.Linear(m, self.instance_classes + 1)
        )

        # 5
        self.iou_score = nn.Linear(m, self.instance_classes + 1)

    def _get_batch_offsets(self, batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def _clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, scale, spatial_shape,
                               rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = softgroup_ops.sec_min(coords, clusters_offset.cuda())
        coords_max = softgroup_ops.sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3, device=self.device)
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3, device=self.device)
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = softgroup_ops.voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = softgroup_ops.voxelization(feats, out_map.cuda())

        voxelization_feats = ME.SparseTensor(features=out_feats, coordinates=out_coords.int().cuda())
        return voxelization_feats, inp_map

    def _forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        output_dict = {}
        """
            Bottom-up Grouping Block
        """
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()]  # (N, m) TODO: the naming p2v is wrong! should be v2p
        semantic_scores = self.semantic_branch(pt_feats)  # (N, nClass), float
        output_dict["semantic_scores"] = semantic_scores
        pt_offsets = self.offset_branch(pt_feats)  # (N, 3), float32
        output_dict["pt_offsets"] = pt_offsets

        if self.current_epoch > self.hparams.cfg.model.prepare_epochs or self.freeze_backbone:
            """
                Top-down Refinement Block
            """
            semantic_scores = semantic_scores.softmax(dim=-1)
            batch_idxs = data_dict["locs_scaled"][:, 0].int()

            # hyperparameters from config
            grouping_radius = self.hparams.cfg.model.grouping_cfg.radius
            grouping_mean_active = self.hparams.cfg.model.grouping_cfg.mean_active
            grouping_num_point_threshold = self.hparams.cfg.model.grouping_cfg.npoint_thr

            class_num_point_mean = torch.tensor(self.hparams.cfg.model.grouping_cfg.class_numpoint_mean,
                                                dtype=torch.float32)
            proposals_offset_list = []
            proposals_idx_list = []

            for class_id in range(self.hparams.cfg.data.classes):
                if class_id in self.hparams.cfg.data.ignore_classes:
                    continue
                scores = semantic_scores[:, class_id].contiguous()
                object_idxs = (scores > self.hparams.cfg.model.grouping_cfg.score_thr).nonzero().view(-1)
                if object_idxs.size(0) < self.hparams.cfg.model.test_cfg.min_npoint:
                    continue
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self._get_batch_offsets(batch_idxs_, batch_size)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]
                idx, start_len = softgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_,
                                                                 grouping_radius, grouping_mean_active)

                proposals_idx, proposals_offset = softgroup_ops.bfs_cluster(class_num_point_mean, idx.cpu(),
                                                                            start_len.cpu(),
                                                                            grouping_num_point_threshold, class_id)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()

                # merge proposals
                if len(proposals_offset_list) > 0:
                    proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                    proposals_offset += proposals_offset_list[-1][-1]
                    proposals_offset = proposals_offset[1:]
                if proposals_idx.size(0) > 0:
                    proposals_idx_list.append(proposals_idx)
                    proposals_offset_list.append(proposals_offset)
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)

            if proposals_offset.shape[0] > self.hparams.cfg.model.train_cfg.max_proposal_num:
                proposals_offset = proposals_offset[:self.hparams.cfg.model.train_cfg.max_proposal_num + 1]
                proposals_idx = proposals_idx[:proposals_offset[-1]]
                assert proposals_idx.shape[0] == proposals_offset[-1]

            output_dict["proposals_idx"] = proposals_idx
            output_dict["proposals_offset"] = proposals_offset

            inst_feats, inst_map = self._clusters_voxelization(
                proposals_idx,
                proposals_offset,
                pt_feats,
                data_dict["locs"],
                rand_quantize=True,
                **self.hparams.cfg.model.instance_voxel_cfg)

            feats = self.tiny_unet(inst_feats)

            # predict mask scores
            mask_scores = self.mask_scoring_branch(feats.features)
            output_dict["mask_scores"] = mask_scores[inst_map.long()]
            output_dict["instance_batch_idxs"] = feats.coordinates[:, 0][inst_map.long()]

            # predict instance cls and iou scores
            feats = self.global_pool(feats)
            output_dict["cls_scores"] = self.classification_branch(feats)
            output_dict["iou_scores"] = self.iou_score(feats)

        return output_dict

    def global_pool(self, x, expand=False):
        indices = x.coordinates[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1,), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = softgroup_ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool
        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    def configure_optimizers(self):
        print("=> configure optimizer...")
        if self.cfg.train.optim.classname == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=self.cfg.train.optim.lr)
        elif self.cfg.train.optim.classname == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                        lr=self.cfg.train.optim.lr,
                                        momentum=self.cfg.train.optim.momentum,
                                        weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented

        return [optimizer]

    def _loss(self, data_dict, output_dict):
        losses = {}
        """semantic loss"""
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda
        sem_seg_criterion = SemSegLoss(self.cfg.data.ignore_label)
        semantic_loss = sem_seg_criterion(output_dict["semantic_scores"], data_dict["sem_labels"])
        losses["semantic_loss"] = semantic_loss

        """offset loss"""
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long
        gt_offsets = data_dict["instance_info"][:, 0:3] - data_dict["locs"]  # (N, 3)
        valid = data_dict["instance_ids"] != self.cfg.data.ignore_label
        pt_offset_criterion = PTOffsetLoss()
        offset_norm_loss, offset_dir_loss = pt_offset_criterion(output_dict["pt_offsets"], gt_offsets, valid_mask=valid)
        valid_count = valid.count_nonzero()
        losses["offset_norm_loss"] = offset_norm_loss
        losses["offset_dir_loss"] = offset_dir_loss

        total_loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + \
                     self.cfg.train.loss_weight[2] * offset_dir_loss

        if self.current_epoch > self.hparams.cfg.model.prepare_epochs:
            proposals_idx = output_dict["proposals_idx"][:, 1].cuda()
            proposals_offset = output_dict["proposals_offset"].cuda()

            # calculate iou of clustered instance
            ious_on_cluster = softgroup_ops.get_mask_iou_on_cluster(proposals_idx, proposals_offset,
                                                                    data_dict["instance_ids"],
                                                                    data_dict["instance_num_point"])

            # filter out background instances
            fg_inds = (data_dict["instance_semantic_cls"] != self.hparams.cfg.data.ignore_label)
            fg_instance_cls = data_dict["instance_semantic_cls"][fg_inds]
            fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

            # overlap > thr on fg instances are positive samples
            max_iou, gt_inds = fg_ious_on_cluster.max(1)
            pos_inds = max_iou >= self.hparams.cfg.model.train_cfg.pos_iou_thr
            pos_gt_inds = gt_inds[pos_inds]

            """classification loss"""

            # follow detection convention: 0 -> K - 1 are fg, K is bg
            labels = fg_instance_cls.new_full((fg_ious_on_cluster.size(0),), self.instance_classes)
            labels[pos_inds] = fg_instance_cls[pos_gt_inds]
            classification_criterion = ClassificationLoss()
            classification_loss = classification_criterion(output_dict["cls_scores"], labels)
            losses["classification_loss"] = classification_loss

            """mask scoring loss"""
            mask_cls_label = labels[output_dict["instance_batch_idxs"].long()]
            slice_inds = torch.arange(0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
            mask_scores_sigmoid_slice = output_dict["mask_scores"].sigmoid()[slice_inds, mask_cls_label]

            mask_label = softgroup_ops.get_mask_label(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_semantic_cls"],
                                                      data_dict["instance_num_point"], ious_on_cluster,
                                                      self.hparams.cfg.model.train_cfg.pos_iou_thr)

            mask_label_weight = (mask_label != -1).float()
            mask_label[mask_label == -1.] = 0.5  # any value is ok
            mask_scoring_criterion = MaskScoringLoss(weight=mask_label_weight, reduction='sum')
            mask_scoring_loss = mask_scoring_criterion(mask_scores_sigmoid_slice, mask_label)
            mask_scoring_loss /= (mask_label_weight.sum() + 1)
            losses["mask_scoring_loss"] = mask_scoring_loss
            """iou scoring loss"""
            ious = softgroup_ops.get_mask_iou_on_pred(proposals_idx, proposals_offset, data_dict["instance_ids"],
                                                      data_dict["instance_num_point"],
                                                      mask_scores_sigmoid_slice.detach())
            fg_ious = ious[:, fg_inds]
            gt_ious, _ = fg_ious.max(1)
            slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
            iou_score_weight = labels < self.instance_classes
            iou_score_slice = output_dict["iou_scores"][slice_inds, labels]
            iou_scoring_criterion = IouScoringLoss(reduction="none")
            iou_scoring_loss = iou_scoring_criterion(iou_score_slice, gt_ious)
            iou_scoring_loss = iou_scoring_loss[iou_score_weight].sum() / (iou_score_weight.count_nonzero() + 1)
            losses["iou_scoring_loss"] = iou_scoring_loss
            total_loss += + self.cfg.train.loss_weight[3] * classification_loss + self.cfg.train.loss_weight[
                4] * mask_scoring_loss + self.cfg.train.loss_weight[5] * iou_scoring_loss

        """total loss"""
        return losses, total_loss

    def _feed(self, data_dict):
        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = softgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"],
                                                              self.cfg.data.mode)  # (M, C), float, cuda
        data_dict = self._forward(data_dict)
        return data_dict

    def training_step(self, data_dict, idx):

        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        self.log("train/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss

    def validation_step(self, data_dict, idx):

        # prepare input and forward
        output_dict = self._feed(data_dict)
        losses, total_loss = self._loss(data_dict, output_dict)

        # log losses
        self.log("val/total_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

        # log semantic prediction accuracy
        semantic_predictions = output_dict["semantic_scores"].max(1)[1].cpu().numpy()
        semantic_accuracy = evaluate_semantic_accuracy(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                       ignore_label=self.hparams.cfg.data.ignore_label)
        semantic_mean_iou = evaluate_semantic_miou(semantic_predictions, data_dict["sem_labels"].cpu().numpy(),
                                                   ignore_label=self.hparams.cfg.data.ignore_label)
        self.log("val_accuracy/semantic_accuracy", semantic_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_accuracy/semantic_mean_iou", semantic_mean_iou, on_step=False, on_epoch=True, sync_dist=True)

        return data_dict, output_dict

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.hparams.cfg.train.clear_cache_every_n_epochs == 0:
            torch.cuda.empty_cache()

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        # evaluate instance predictions
        if self.current_epoch > self.hparams.cfg.model.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            for batch, output in outputs:
                pred_instances = self._get_instances(output["proposals_idx"].cpu(), output["semantic_scores"].cpu(),
                                                     output["cls_scores"].cpu(), output["iou_scores"].cpu(),
                                                     output["mask_scores"].cpu())
                gt_instances = self._get_gt_instances(batch["sem_labels"].cpu(), batch["instance_ids"].cpu())
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            evaluator = ScanNetEval(self.hparams.cfg.data.class_names)
            evaluation_result = evaluator.evaluate(all_pred_insts, all_gt_insts)
            self.log("val_accuracy/AP", evaluation_result["all_ap"], sync_dist=True)
            self.log("val_accuracy/AP_50", evaluation_result['all_ap_50%'], sync_dist=True)
            self.log("val_accuracy/AP_25", evaluation_result["all_ap_25%"], sync_dist=True)

    def test_step(self, data_dict, idx):

        # prepare input and forward
        output_dict = self._feed(data_dict)

        return output_dict

    def predict_step(self, data_dict, idx, dataloader_idx):
        # prepare input and forward
        data_dict = self._feed(data_dict)

        # semantic prediction
        semantic_predictions = data_dict["semantic_scores"].max(1)[1]

        if self.current_epoch > self.hparams.cfg.model.prepare_epochs:
            # instance prediction
            pred_instances = self._get_instances(data_dict["proposals_idx"], data_dict["semantic_scores"],
                                                 data_dict["cls_scores"], data_dict["iou_scores"],
                                                 data_dict["mask_scores"])

        # save predictions
        return data_dict

    def _get_instances(self, proposals_idx, semantic_scores, cls_scores, iou_scores, mask_scores):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)
        cls_scores = cls_scores.softmax(1)
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        for i in range(self.instance_classes):
            cls_pred = cls_scores.new_full((num_instances,), i + 1, dtype=torch.long)
            cur_cls_scores = cls_scores[:, i]
            cur_iou_scores = iou_scores[:, i]
            cur_mask_scores = mask_scores[:, i]
            score_pred = cur_cls_scores * cur_iou_scores.clamp(0, 1)
            mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int, device="cpu")
            mask_inds = cur_mask_scores > self.hparams.cfg.model.test_cfg.mask_score_thr
            cur_proposals_idx = proposals_idx[mask_inds].long()
            mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

            # filter low score instance
            inds = cur_cls_scores > self.hparams.cfg.model.test_cfg.cls_score_thr
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]

            # filter too small instances
            npoint = mask_pred.sum(1)
            inds = npoint >= self.hparams.cfg.model.test_cfg.min_npoint
            cls_pred = cls_pred[inds]
            score_pred = score_pred[inds]
            mask_pred = mask_pred[inds]
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.append(mask_pred)
        cls_pred = torch.cat(cls_pred_list).numpy()
        score_pred = torch.cat(score_pred_list).numpy()
        mask_pred = torch.cat(mask_pred_list).numpy()

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            instances.append(pred)
        return instances

    def _get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        label_shift = len(self.hparams.cfg.data.ignore_classes)
        semantic_labels = semantic_labels - label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins
        return gt_ins
