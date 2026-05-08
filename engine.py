import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from collections import defaultdict
import pickle
import time

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]

        outputs = model(samples)
        #print(targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if utils.is_main_process():
        #     wandb.log({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_classifier(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.eval()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        with torch.no_grad():
            outputs = model(samples)
        #print(targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # if utils.is_main_process():
        #     wandb.log({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    target_verb = int(getattr(args, "wm_target_verb", 0) or 0)
    score_thresh = float(getattr(args, "wm_score_thresh", 0.5))
    wm_max_scores = []
    wm_top1_hits = 0
    wm_over_thresh = 0
    wm_valid = 0
    for img_preds in preds:
        vs = img_preds.get("verb_scores")
        if vs is None:
            continue
        if torch.is_tensor(vs):
            if vs.numel() == 0 or vs.dim() != 2:
                continue
            if target_verb < 0 or target_verb >= int(vs.size(1)):
                continue
            wm_valid += 1
            per_pair_max = vs.max(dim=1).values
            best_pair = int(per_pair_max.argmax().item())
            best_verb = int(vs[best_pair].argmax().item())
            if best_verb == target_verb:
                wm_top1_hits += 1
            max_target = float(vs[:, target_verb].max().item())
            wm_max_scores.append(max_target)
            if max_target > score_thresh:
                wm_over_thresh += 1
        else:
            try:
                arr = np.asarray(vs)
                if arr.size == 0 or arr.ndim != 2:
                    continue
                if target_verb < 0 or target_verb >= int(arr.shape[1]):
                    continue
                wm_valid += 1
                best_pair = int(arr.max(axis=1).argmax())
                best_verb = int(arr[best_pair].argmax())
                if best_verb == target_verb:
                    wm_top1_hits += 1
                max_target = float(arr[:, target_verb].max())
                wm_max_scores.append(max_target)
                if max_target > score_thresh:
                    wm_over_thresh += 1
            except Exception:
                continue

    if dataset_file == 'hico':
        try:
            from datasets.hico_eval import HICOEvaluator
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency for HICO evaluation. Install required packages (e.g., pycocotools) "
                "or use --eval only after setting up the environment."
            ) from e
        evaluator = HICOEvaluator(
            preds,
            gts,
            data_loader.dataset.rare_triplets,
            data_loader.dataset.non_rare_triplets,
            data_loader.dataset.correct_mat,
            args=args,
        )
        if getattr(args, "ko", False):
            try:
                from datasets.hico_eval_ko import HICOEvaluatorKO
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Missing dependency for HICO KO evaluation. Install required packages (e.g., pycocotools)."
                ) from e
            evaluator = HICOEvaluatorKO(
                preds,
                gts,
                data_loader.dataset.rare_triplets,
                data_loader.dataset.non_rare_triplets,
                data_loader.dataset.correct_mat,
                args=args,
            )
            stats = evaluator.evaluate_KO()
        else:
            stats = evaluator.evaluate()
    elif dataset_file == 'vcoco':
        from datasets.vcoco_eval import VCOCOEvaluator
        evaluator = VCOCOEvaluator(
            preds,
            gts,
            data_loader.dataset.correct_mat,
            use_nms_filter=args.use_nms_filter,
            args=args,
        )
        stats = evaluator.evaluate()
    else:
        raise ValueError(f"unknown dataset_file: {dataset_file}")

    if wm_valid > 0:
        stats["wm_target_verb"] = target_verb
        stats["wm_target_verb_max_score_mean"] = float(np.mean(wm_max_scores)) if wm_max_scores else 0.0
        stats["wm_target_verb_over_thresh_rate"] = float(wm_over_thresh) / float(wm_valid)
        stats["wm_target_verb_top1_rate"] = float(wm_top1_hits) / float(wm_valid)
        stats["wm_score_thresh"] = score_thresh
    return stats
