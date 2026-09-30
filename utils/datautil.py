import os
import torch
import numpy as np
from PIL import Image
import cv2
import av
from typing import List, Optional
dtype = torch.FloatTensor

def read_yuv_video(yuv_filename, width, height, frame_count, color_format='420'):
    if color_format == '420':
        frame_size = (width * height) + (width // 2 * height // 2 * 2)
    with open(yuv_filename, 'rb') as f:
        raw_data = f.read()
    video_data = np.zeros((height, width, frame_count), dtype=np.float32)
    # 逐帧读取并解析 YUV 数据
    for frame_idx in range(frame_count):
        start_idx = frame_idx * frame_size
        end_idx = start_idx + frame_size
        # 读取当前帧的 YUV 数据
        frame_data = raw_data[start_idx:end_idx]
        # 提取 Y, U, V 分量
        y_data = frame_data[:width * height]
        uv_data = frame_data[width * height:]
        # 将 Y 分量转换为二维数组
        y_frame = np.frombuffer(y_data, dtype=np.uint8).reshape(height, width)
        # 将 U, V 分量转换为二维数组
        # if color_format == '420':
        #     u_data = uv_data[:width * height // 4]
        #     v_data = uv_data[width * height // 4:]
        #     u_frame = np.frombuffer(u_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     v_frame = np.frombuffer(v_data, dtype=np.uint8).reshape(height // 2, width // 2)
        #     u_frame = np.repeat(np.repeat(u_frame, 2, axis=0), 2, axis=1)
        #     v_frame = np.repeat(np.repeat(v_frame, 2, axis=0), 2, axis=1)
        # 将 Y, U, V 分量合并为 RGB 图像（可选）
        # 你可以使用 OpenCV 或其他库将 YUV 转换为 RGB 但是这里只保留 Y 分量
        video_data[:, :, frame_idx] = y_frame / 255.0
    return video_data


def read_y4m_video(y4m_filename: str, frame_count: Optional[int] = None):
    video_data = []
    read_frame_num = 0
    
    try: 
        container = av.open(y4m_filename)
        
        # 验证视频流存在
        if not container.streams.video:
            raise ValueError("文件中未找到视频流")
        
        # 获取视频流信息
        video_stream = container.streams.video[0]
        orig_width = video_stream.width
        orig_height = video_stream.height
        # print(f"原始视频分辨率: {orig_width}x{orig_height}")
        
        # 检查指定的分辨率是否合法
        target_size = (orig_width, orig_height)
        
        # 逐帧解码
        for frame in container.decode(video=0):
            # 达到指定帧数则停止
            if frame_count is not None and read_frame_num >= frame_count:
                break
            
            # 转换为RGB格式的numpy数组
            img_rgb = frame.to_ndarray(format='rgb24')
            
            # 添加到结果列表
            video_data.append(img_rgb)
            read_frame_num += 1
        
        print(f"成功读取 {read_frame_num} 帧")
        
    except Exception as e:
        print(f"读取Y4M文件时出错: {str(e)}")
        raise
    finally: # 确保资源正确释放
        try:
            container.close()
        except:
            pass

    video_data = np.stack(video_data, axis=2) #形状为(H, W, 10, 3)
    video_data = video_data / 255.0

    return video_data

def generate_random_mask_3d(H, W, C, visible_ratio=0.1):
    num_visible = int(H * W * C * visible_ratio)
    all_positions = np.arange(H * W * C)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W * C, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, C)
    mask = torch.from_numpy(mask).type(dtype)
    return mask

def generate_random_mask_4d(H, W, N, C, visible_ratio=0.1):
    num_visible = int(H * W * N * C * visible_ratio)
    all_positions = np.arange(H * W * N * C)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W * N * C, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W, N, C)
    mask = torch.from_numpy(mask).type(dtype)
    return mask

def generate_random_mask(H, W, visible_ratio=0.1):
    num_visible = int(H * W * visible_ratio)
    all_positions = np.arange(H * W)
    visible_positions = np.random.choice(all_positions, size=num_visible, replace=False)
    mask = np.zeros(H * W, dtype=np.uint8)
    mask[visible_positions] = 1
    mask = mask.reshape(H, W)
    # mask = np.repeat(mask, 3, axis=2)
    mask = torch.from_numpy(mask).type(dtype)
    return mask

def load_grayscale_images_from_directory(directory):
    png_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])
    images = []
    for file in png_files:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path)  # 读取为灰度图像
        image_array = np.array(image)  # 转换为 NumPy 数组
        images.append(image_array)
    if len(images) > 0:
        max_value = 2 ** 16 - 1
        stacked_images = np.stack(images, axis=-1).astype(np.float32) / max_value
        return stacked_images
    else:
        return None

def load_LF_images_from_directory(directory, length=30):
    png_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])[:length]
    images = []
    for file in png_files:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path)  # 读取为灰度图像
        image_array = np.array(image)  # 转换为 NumPy 数组
        images.append(image_array)
    if len(images) > 0:
        stacked_images = np.stack(images, axis=-1).astype(np.float32) / 255.0
        return stacked_images
    else:
        return None

def truncated_linear_stretch(image, truncated_value=0, maxout=1, minout=0):
    def gray_process(gray, maxout, minout):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = (gray - truncated_down) / ((truncated_up - truncated_down) / (maxout - minout))
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.float32(gray_new)

    height, width, band = image.shape
    out =np.zeros((height, width, band))
    
    for b in range(band):
        out[:,:,b] = gray_process(image[:,:,b], maxout, minout)
    return out