import math
import torch
from numbers import Real

def gaussian_repar(mu, sigma):
    noise = torch.randn_like(mu) * sigma
    return mu + noise


def gaussian_log_prob(x, x_mu, x_sigma):
    if isinstance(x_sigma, Real):
        log_x_sigma = math.log(x_sigma)
    else:
        log_x_sigma = torch.log(x_sigma)

    log_prob = - 0.5 * math.log(2 * math.pi) - log_x_sigma \
               - 0.5 * (torch.pow(x - x_mu, 2) / x_sigma ** 2)
    return log_prob #size [128, 5]

def reconstruct_to_tensor_4D_tube(rec_X, batch_coords, original_shape, C=3):
    H, W, N = original_shape
    device = rec_X.device
    # 初始化目标张量
    reconstructed_tensor = torch.zeros(H, W, N, C, device=device)
    count_tensor = torch.zeros(H, W, N, device=device)
    # 将坐标转换为long类型
    coords = batch_coords.long()  # [N, 2]
    # 确保坐标在合法范围内
    coords[:, 0] = torch.clamp(coords[:, 0], 0, H-1)
    coords[:, 1] = torch.clamp(coords[:, 1], 0, W-1)
    coords[:, 2] = torch.clamp(coords[:, 2], 0, N-1)
    # 使用scatter_add_进行高效填充
    reconstructed_tensor[coords[:, 0], coords[:, 1], coords[:, 2], :] += rec_X.squeeze()
    count_tensor[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
    return reconstructed_tensor, count_tensor

def reconstruct_to_tensor_4D(rec_X, batch_coords, original_shape):
    H, W, N, C = original_shape
    device = rec_X.device
    # 初始化目标张量
    reconstructed_tensor = torch.zeros(H, W, N, C, device=device)
    count_tensor = torch.zeros(H, W, N, C, device=device)
    # 将坐标转换为long类型
    coords = batch_coords.long()  # [N, 2]
    # 确保坐标在合法范围内
    coords[:, 0] = torch.clamp(coords[:, 0], 0, H-1)
    coords[:, 1] = torch.clamp(coords[:, 1], 0, W-1)
    coords[:, 2] = torch.clamp(coords[:, 2], 0, N-1)
    coords[:, 3] = torch.clamp(coords[:, 3], 0, C-1)
    # 使用scatter_add_进行高效填充
    reconstructed_tensor[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] += rec_X.squeeze()
    count_tensor[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] += 1
    return reconstructed_tensor, count_tensor

def reconstruct_to_tensor_6D(rec_X, batch_coords, original_shape):
    H1, H2, W1, W2, N, C = original_shape
    device = rec_X.device
    # 初始化目标张量
    reconstructed_tensor = torch.zeros(H1, H2, W1, W2, N, C, device=device)
    count_tensor = torch.zeros(H1, H2, W1, W2, N, C, device=device)
    # 将坐标转换为long类型
    coords = batch_coords.long()  # [N, 2]
    # 确保坐标在合法范围内
    coords[:, 0] = torch.clamp(coords[:, 0], 0, H1-1)
    coords[:, 1] = torch.clamp(coords[:, 1], 0, H2-1)
    coords[:, 2] = torch.clamp(coords[:, 2], 0, W1-1)
    coords[:, 3] = torch.clamp(coords[:, 3], 0, W2-1)
    coords[:, 4] = torch.clamp(coords[:, 4], 0, N-1)
    coords[:, 5] = torch.clamp(coords[:, 5], 0, C-1)
    # 使用scatter_add_进行高效填充
    reconstructed_tensor[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 4], coords[:, 5]] += rec_X.squeeze()
    count_tensor[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 4], coords[:, 5]] += 1
    return reconstructed_tensor, count_tensor

def reconstruct_to_tensor_3D(rec_X, batch_coords, original_shape):
    H, W, C = original_shape
    device = rec_X.device
    # 初始化目标张量
    reconstructed_tensor = torch.zeros(H, W, C, device=device)
    count_tensor = torch.zeros(H, W, C, device=device)
    # 将坐标转换为long类型
    coords = batch_coords.long()  # [N, 3]
    # 确保坐标在合法范围内
    coords[:, 0] = torch.clamp(coords[:, 0], 0, H-1)
    coords[:, 1] = torch.clamp(coords[:, 1], 0, W-1)
    coords[:, 2] = torch.clamp(coords[:, 2], 0, C-1)
    # 使用scatter_add_进行高效填充
    reconstructed_tensor[coords[:, 0], coords[:, 1], coords[:, 2]] += rec_X.squeeze()
    count_tensor[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
    return reconstructed_tensor, count_tensor

def reconstruct_to_tensor_2D_tube(rec_X, batch_coords, original_shape, C=3):
    H, W = original_shape
    device = rec_X.device
    # 初始化目标张量
    reconstructed_tensor = torch.zeros(H, W, C, device=device)
    count_tensor = torch.zeros(H, W, device=device)
    # 将坐标转换为long类型
    coords = batch_coords.long()  # [N, 2]
    # 确保坐标在合法范围内
    coords[:, 0] = torch.clamp(coords[:, 0], 0, H-1)
    coords[:, 1] = torch.clamp(coords[:, 1], 0, W-1)
    # 使用scatter_add_进行高效填充
    reconstructed_tensor[coords[:, 0], coords[:, 1], :] += rec_X.squeeze()
    count_tensor[coords[:, 0], coords[:, 1]] += 1
    return reconstructed_tensor, count_tensor