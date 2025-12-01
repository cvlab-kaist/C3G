import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
import pdb

KEY_OUTPUT = 'metric_depth'

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    if mask is None:
        mask = torch.ones_like(target)

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask=None, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        if mask is not None:
            assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            scale, shift = scale.detach(), shift.detach()

            scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            loss = nn.functional.l1_loss(scaled_prediction[mask], target[mask])
        
        else:
            assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."
            scale, shift = compute_scale_and_shift(prediction, target, mask=None)
            scale, shift = scale.detach(), shift.detach()

            scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
            loss = nn.functional.l1_loss(scaled_prediction, target)

        if not return_interpolated:
            return loss, scale, shift
        return loss, scale, shift

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-6))
    return mag, angle


def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]


class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        loss = nn.functional.l1_loss(grad_pred[0], grad_gt[0])
        loss = loss + \
            nn.functional.l1_loss(grad_pred[1], grad_gt[1])
        if not return_interpolated:
            return loss
        return loss, intr_input
    
class ScaleAndShiftInvariantGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSILoss"

    def forward(self, prediction, target, mask=None, interpolate=True, return_interpolated=False):
        
        if prediction.shape[-1] != target.shape[-1] and interpolate:
            prediction = nn.functional.interpolate(prediction, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = prediction
        else:
            intr_input = prediction

        if mask is not None:
            prediction, target, mask = prediction.squeeze(), target.squeeze(), mask.squeeze()
            prediction, target, mask = prediction.unsqueeze(dim=0), target.unsqueeze(dim=0), mask.unsqueeze(dim=0)
            assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            scale, shift = scale.detach(), shift.detach()
        else:
            assert prediction.shape == target.shape, f"Shape mismatch: Expected same shape but got {prediction.shape} and {target.shape}."
            scale, shift = compute_scale_and_shift(prediction, target, mask=None)
            scale, shift = scale.detach(), shift.detach()

        scaled_prediction = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        grad_gt = grad(target)
        grad_pred = grad(scaled_prediction)
        # mask_g = grad_mask(mask)


        loss = nn.functional.l1_loss(scaled_prediction, target)

        loss = loss + 0.001 * nn.functional.l1_loss(grad_pred[0], grad_gt[0])
        loss = loss + \
            0.001 * nn.functional.l1_loss(grad_pred[1], grad_gt[1])
        if not return_interpolated:
            return loss, scale, shift
        return loss, scale, shift