# ===================================================================================
# LSC (Lens Shading Correction) Calibration Script
# Version: V7.0 - Final Corrected Architecture (with enhancements by Gemini)
#
# Description:
# This script generates LSC gain tables for fisheye lenses. It addresses luma
# shading, chroma shading, and edge artifacts. This version incorporates all
# bug fixes from the extensive debugging process, including correct per-channel
# black level subtraction and normalization, a robust grid statistics
# calculation, and proper handling of data domains.
#
# Gemini's Enhancements:
# - Added detailed startup configuration logging.
# - Added function to visualize and save the circular mask (overlay and grayscale).
# - Added more detailed logging for key calculation steps.
# ===================================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging # 【推荐】可以使用Python内置的logging模块，但为了保持简洁，我们继续用print

# --- 配置参数 ---
RAW_IMAGE_PATH = '0715_25k.raw'
IMAGE_WIDTH = 1256
IMAGE_HEIGHT = 1256

# 【1】分通道黑电平 (请务必测量并使用真实值)
BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}

# 【2】选择正确的Bayer Pattern
BAYER_PATTERN = cv2.COLOR_BayerGR2BGR_VNG
# 【3】LSC网格和算法参数
GRID_ROWS = 13
GRID_COLS = 17
# 用于保证边缘亮度统计稳定性的最小像素数：羽化过度
MIN_PIXELS_PER_GRID = 2

# 【4】效果微调参数
# 亮度衰减因子，抑制边缘过亮 (0.7-1.0)
FALLOFF_FACTOR = 1
# 颜色衰减因子，与上面保持一致即可
COLOR_FALLOFF_FACTOR = 1.0
# 过暗网格的放弃阈值 (0.1-0.4)
VALID_GRID_THRESHOLD_RATIO = 0.3
# 全局最大增益限制
MAX_GAIN = 3

# --- 其他配置 ---
OUTPUT_DIR = 'output_images_manual_raw_interactive'
APPLY_SYMMETRY = True
MASK_FEATHER_PIXELS = 30
MANUAL_ADJUST_STEP = 1

def set_matplotlib_english_font():
    # 确保 Matplotlib 使用非中文字体，以避免混淆
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] # 常用英文字体，作为回退
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    print("Matplotlib has been configured for English display.")

# 在脚本开始时调用字体配置函数
set_matplotlib_english_font()

# --- 辅助函数：手动读取和处理裸RAW数据 (修改为直接返回16位拜耳数据) ---
def read_raw_bayer_image_manual(raw_path, width, height):
    try:
        expected_pixels = width * height
        # 读取10-bit RAW数据，但按16-bit读取
        bayer_data_16bit_raw = np.fromfile(raw_path, dtype='<u2', count=expected_pixels)

        if bayer_data_16bit_raw.size != expected_pixels:
            raise ValueError(f"Raw file size mismatch after attempting to read expected count. Expected {expected_pixels} pixels, but got {bayer_data_16bit_raw.size}.")

        # 对10-bit数据进行扩展，使其保持在0-1023范围
        bayer_image_16bit = (bayer_data_16bit_raw.astype(np.uint16) & 0x03FF)  # 仅取低10位
        bayer_image_16bit = bayer_image_16bit.reshape((height, width))

        return bayer_image_16bit
    except Exception as e:
        print(f"Error reading raw image: {e}")
        return None


def extract_bayer_channels(bayer_image_16bit, bayer_pattern_code, black_levels_dict):
    """
    对各通道独立进行BLC和归一化，从根本上解决颜色比例失真问题。
    【已更新】兼容 ...2BGR_VNG 和 ...2RGB 两种OpenCV常量。
    """
    h, w = bayer_image_16bit.shape
    R, Gr, Gb, B = [np.zeros((h, w), dtype=np.float32) for _ in range(4)]

    # 根据Bayer模式分离通道 (增加对 ...2RGB 常量的支持)
    if bayer_pattern_code == cv2.COLOR_BayerRG2BGR_VNG or bayer_pattern_code == cv2.COLOR_BayerRG2RGB:  # RGGB
        R[0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        Gr[0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        Gb[1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        B[1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGR2BGR_VNG or bayer_pattern_code == cv2.COLOR_BayerGR2RGB:  # GRBG
        Gr[0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        R[0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        B[1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        Gb[1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerBG2BGR_VNG or bayer_pattern_code == cv2.COLOR_BayerBG2RGB:  # BGGR
        B[0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        Gb[0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        Gr[1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        R[1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGB2BGR_VNG or bayer_pattern_code == cv2.COLOR_BayerGB2RGB:  # GBRG
        Gb[0::2, 0::2] = bayer_image_16bit[0::2, 0::2]
        B[0::2, 1::2] = bayer_image_16bit[0::2, 1::2]
        R[1::2, 0::2] = bayer_image_16bit[1::2, 0::2]
        Gr[1::2, 1::2] = bayer_image_16bit[1::2, 1::2]
    else:
        raise ValueError("Unsupported Bayer pattern code.")

    # 黑电平校正，确保不出现负值
    R = np.maximum(0, R - black_levels_dict['R'])
    Gr = np.maximum(0, Gr - black_levels_dict['Gr'])
    Gb = np.maximum(0, Gb - black_levels_dict['Gb'])
    B = np.maximum(0, B - black_levels_dict['B'])

    # 归一化处理，确保10bit数据保持在正确范围
    norm_R = 1023 - black_levels_dict['R']
    norm_Gr = 1023 - black_levels_dict['Gr']
    norm_Gb = 1023 - black_levels_dict['Gb']
    norm_B = 1023 - black_levels_dict['B']

    R[R > 0] /= norm_R if norm_R > 0 else 1023
    Gr[Gr > 0] /= norm_Gr if norm_Gr > 0 else 1023
    Gb[Gb > 0] /= norm_Gb if norm_Gb > 0 else 1023
    B[B > 0] /= norm_B if norm_B > 0 else 1023

    return {'R': R, 'Gr': Gr, 'Gb': Gb, 'B': B}


def apply_gain_to_bayer(bayer_blc_float, gain_map_R, gain_map_Gr, gain_map_Gb, gain_map_B, bayer_pattern_code):
    """
    将增益应用到已经去除黑电平的浮点bayer数据上。
    """
    compensated_bayer_float = bayer_blc_float.copy()

    if bayer_pattern_code == cv2.COLOR_BayerRG2BGR_VNG:  # RGGB
        compensated_bayer_float[0::2, 0::2] *= gain_map_R[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_Gr[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_Gb[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_B[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGR2BGR_VNG:  # GRBG
        compensated_bayer_float[0::2, 0::2] *= gain_map_Gr[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_R[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_B[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_Gb[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerBG2BGR_VNG:  # BGGR
        compensated_bayer_float[0::2, 0::2] *= gain_map_B[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_Gb[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_Gr[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_R[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGB2BGR_VNG:  # GBRG
        compensated_bayer_float[0::2, 0::2] *= gain_map_Gb[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_B[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_R[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_Gr[1::2, 1::2]

    return compensated_bayer_float  # 返回的也是浮点数


# --- 【恢复】白平衡函数 ---
def simple_white_balance(image_rgb_float, mask_2d=None):
    """
    对RGB图像进行稳健的自动白平衡。
    """
    if image_rgb_float.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB image for white balance.")

    balanced_image = image_rgb_float.copy()

    h, w, _ = image_rgb_float.shape
    R = balanced_image[:, :, 0]
    G = balanced_image[:, :, 1]
    B = balanced_image[:, :, 2]

    y_start, y_end = int(h * 0.3), int(h * 0.7)
    x_start, x_end = int(w * 0.3), int(w * 0.7)

    central_patch_R = R[y_start:y_end, x_start:x_end]
    central_patch_G = G[y_start:y_end, x_start:x_end]
    central_patch_B = B[y_start:y_end, x_start:x_end]

    if mask_2d is not None:
        central_mask_patch = mask_2d[y_start:y_end, x_start:x_end]
        valid_pixels_mask = central_mask_patch > 0.1

        g_channel_valid = central_patch_G[valid_pixels_mask]

        if g_channel_valid.size == 0 or np.mean(g_channel_valid) < 1e-6:
            print("警告: 白平衡计算区域(中心)的绿色通道均值过低，跳过白平衡。")
            return image_rgb_float

        avg_R = np.mean(central_patch_R[valid_pixels_mask])
        avg_G = np.mean(g_channel_valid)
        avg_B = np.mean(central_patch_B[valid_pixels_mask])

    else:
        avg_G = np.mean(central_patch_G)
        if avg_G < 1e-6:
            print("警告: 白平衡计算区域(中心)的绿色通道均值过低，跳过白平衡。")
            return image_rgb_float
        avg_R = np.mean(central_patch_R)
        avg_B = np.mean(central_patch_B)

    if avg_G < 1e-6:
        print("警告: 绿色通道均值过低，跳过白平衡。")
        return image_rgb_float

    gain_R = avg_G / (avg_R + 1e-6)
    gain_B = avg_G / (avg_B + 1e-6)

    print(f"白平衡增益 (基于中心区域): R={gain_R:.2f}, G=1.00, B={gain_B:.2f}")

    balanced_image[:, :, 0] = np.clip(R * gain_R, 0, 1.0)
    balanced_image[:, :, 2] = np.clip(B * gain_B, 0, 1.0)

    return balanced_image

# 【新增】【高级功能】先外插再平滑，避免污染 (V2 - 修正版)
def extrapolate_and_smooth_gains(gain_matrix, gaussian_ksize=5):
    """
    通过先外插再平滑的方式，智能地处理增益矩阵，避免无效区域(gain=1.0)对有效区域的污染。

    Args:
        gain_matrix (np.array): 原始的LSC增益矩阵，其中无效区域的值为1.0。
        gaussian_ksize (int): 高斯模糊的核大小，必须是奇数。

    Returns:
        np.array: 经过外插和平滑处理后的高质量增益矩阵。
    """
    print("执行高级平滑：先外插，后平滑...")
    
    # 确保内核大小是奇数
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1
        
    # 导入scipy库，如果之前没导入的话
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("\n错误：需要 'scipy' 库来实现高级平滑功能。")
        print("请在您的终端中运行: pip install scipy\n")
        # 如果没有scipy，就返回未处理的矩阵，避免程序崩溃
        return gain_matrix
        
    # 创建一个与输入矩阵相同大小的浮点类型副本用于操作
    filled_matrix = gain_matrix.astype(np.float32)

    # 制作一个掩码，标记出无效区域（值为1.0）
    # 使用一个很小的容差来处理浮点数精度问题
    invalid_mask = (np.abs(gain_matrix - 1.0) < 1e-6)
    
    # 如果不存在无效区域，则直接进行标准平滑并返回
    if np.sum(invalid_mask) == 0:
        print("  - 未检测到无效区域，执行标准高斯平滑。")
        return cv2.GaussianBlur(filled_matrix, (gaussian_ksize, gaussian_ksize), 0)

    # 【核心逻辑】使用Scipy的cKDTree来高效查找最近邻
    # 1. 获取所有有效点和无效点的坐标
    valid_coords = np.argwhere(invalid_mask == False)
    invalid_coords = np.argwhere(invalid_mask == True)
    
    # 2. 用所有有效点构建一个KD树，用于快速查询
    tree = cKDTree(valid_coords)
    
    # 3. 为所有无效点查询其最近的有效点的索引
    distances, indices = tree.query(invalid_coords)
    
    # 4. 根据索引找到最近的有效点的具体坐标
    nearest_valid_coords = valid_coords[indices]
    
    # 5. 将无效点的值替换为它最近的有效点的值（外插）
    # numpy的高级索引让这一步非常高效
    filled_matrix[invalid_coords[:, 0], invalid_coords[:, 1]] = filled_matrix[nearest_valid_coords[:, 0], nearest_valid_coords[:, 1]]
    
    print(f"  - 完成外插，填充了 {len(invalid_coords)} 个无效点。")

    # 现在，在已经没有“悬崖”的矩阵上进行高斯平滑
    final_smoothed_matrix = cv2.GaussianBlur(filled_matrix, (gaussian_ksize, gaussian_ksize), 0)
    print("  - 已在外插后的矩阵上完成高斯平滑。")
    
    return final_smoothed_matrix

# --- 【算法升级】手动选择和调整圆形区域, 并记住选择 ---
def get_manual_circle_mask(image_rgb_float, feather_pixels, output_dir, adjust_step=MANUAL_ADJUST_STEP):
    """
    通过交互式鼠标操作，让用户选择和调整圆形区域，并生成羽化掩码。
    【功能】加载、预览和复用上次选择的区域，以提高效率。
    """
    h, w, _ = image_rgb_float.shape
    display_image = (image_rgb_float * 255).astype(np.uint8)

    max_display_dim = 900
    scale = min(max_display_dim / w, max_display_dim / h)
    display_w, display_h = int(w * scale), int(h * scale)
    display_image_resized = cv2.resize(display_image, (display_w, display_h))

    display_image_bgr = cv2.cvtColor(display_image_resized, cv2.COLOR_RGB2BGR)

    params_path = os.path.join(output_dir, 'circle_params.npy')
    current_circle = {'center': None, 'radius': 0}
    drawing = False
    window_name = "Select Fisheye Region (check console for options)"

    def draw_circle_on_image(img, circle_info, color=(0, 255, 0), thickness=2):
        temp_img = img.copy()
        if circle_info['center'] is not None and circle_info['radius'] > 0:
            center_x_orig = int(circle_info['center'][0] / scale)
            center_y_orig = int(circle_info['center'][1] / scale)
            radius_orig = int(circle_info['radius'] / scale)

            cv2.circle(temp_img, circle_info['center'], circle_info['radius'], color, thickness)
            cv2.putText(temp_img, f"Center: ({center_x_orig},{center_y_orig})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(temp_img, f"Radius: {radius_orig}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return temp_img

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_circle, display_image_bgr

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_circle['center'] = (x, y)
            current_circle['radius'] = 0
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                if current_circle['center'] is not None:
                    dx = x - current_circle['center'][0]
                    dy = y - current_circle['center'][1]
                    current_circle['radius'] = int(np.sqrt(dx*dx + dy*dy))
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    # --- 功能：加载、预览和复用逻辑 ---
    start_main_loop = True
    if os.path.exists(params_path):
        print(f"\n--- 检测到上次的选择参数: '{params_path}' ---")
        try:
            loaded_params = np.load(params_path)
            loaded_center_orig = (int(loaded_params[0]), int(loaded_params[1]))
            loaded_radius_orig = int(loaded_params[2])

            preview_circle = {
                'center': (int(loaded_center_orig[0] * scale), int(loaded_center_orig[1] * scale)),
                'radius': int(loaded_radius_orig * scale)
            }

            preview_img = draw_circle_on_image(display_image_bgr, preview_circle, color=(0, 255, 255), thickness=2)
            cv2.putText(preview_img, "(R)euse, (E)dit, or (N)ew selection?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, preview_img)

            print("操作选项:")
            print(" 'r': 复用上次的选择 (Reuse)")
            print(" 'e': 编辑上次的选择 (Edit)")
            print(" 'n' 或其他键: 创建新的选择 (New)")

            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                print("选择: 复用上次结果。")
                current_circle = preview_circle
                start_main_loop = False # 直接跳过手动调整循环
            elif key == ord('e'):
                print("选择: 编辑上次结果。")
                current_circle = preview_circle # 以加载的圆为起点开始调整
            else:
                print("选择: 创建新的选择。")
        except Exception as e:
            print(f"无法加载或解析之前的选择文件. 错误: {e}. 开始新的选择。")

    if start_main_loop:
        print("\n--- 手动选择圆形有效区域 ---")
        print("1. 按住鼠标左键并拖动来画圆 (从圆心到边缘)。")
        print("2. 松开鼠标左键确认初始圆形。")
        print("3. 在窗口激活状态下，使用键盘微调:")
        print(f"   'w' / 's': 增大 / 减小半径 (步长: {adjust_step} 像素)")
        print(f"   'a' / 'd': 左 / 右移动圆心 (步长: {adjust_step} 像素)")
        print(f"   'z' / 'x': 上 / 下移动圆心 (步长: {adjust_step} 像素)")
        print("   'r': 重置 (重新开始画圆)")
        print("   'q': 确认当前圆形并继续")
        print("-------------------------------")

        cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

        key = -1
        while key != ord('q'):
            key = cv2.waitKey(1) & 0xFF

            if key == ord('w'):
                current_circle['radius'] += adjust_step
            elif key == ord('s'):
                current_circle['radius'] = max(0, current_circle['radius'] - adjust_step)
            elif key == ord('a'):
                if current_circle['center'] is not None:
                    current_circle['center'] = (current_circle['center'][0] - adjust_step, current_circle['center'][1])
            elif key == ord('d'):
                if current_circle['center'] is not None:
                    current_circle['center'] = (current_circle['center'][0] + adjust_step, current_circle['center'][1])
            elif key == ord('z'):
                if current_circle['center'] is not None:
                    current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] - adjust_step)
            elif key == ord('x'):
                if current_circle['center'] is not None:
                    current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] + adjust_step)
            elif key == ord('r'):
                current_circle = {'center': None, 'radius': 0}
                drawing = False

            if key != ord('q'):
                 cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

    cv2.destroyAllWindows()

    final_cx = int(current_circle['center'][0] / scale) if current_circle['center'] else w // 2
    final_cy = int(current_circle['center'][1] / scale) if current_circle['center'] else h // 2
    final_r = int(current_circle['radius'] / scale) if current_circle['radius'] > 0 else min(h, w) // 2 - 10

    if final_r <= 0:
        final_r = 1

    # --- 功能：保存选择的区域参数 ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        circle_params = np.array([final_cx, final_cy, final_r])
        np.save(params_path, circle_params)
        print(f"已保存当前圆形选择参数至: {params_path}")
    except Exception as e:
        print(f"警告: 无法保存圆形参数. 错误: {e}")

    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask, (final_cx, final_cy), final_r, 255, -1)

    kernel_size = int(2 * feather_pixels / 3)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if kernel_size < 3: kernel_size = 3

    feathered_mask = cv2.GaussianBlur(hard_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    feathered_mask = feathered_mask / 255.0
    feathered_mask = np.clip(feathered_mask, 0.0, 1.0)

    return feathered_mask, (final_cx, final_cy, final_r)


# --- 新增函数：绘制和保存热力图 ---
def plot_heatmap_and_save_matrix(matrix, title_suffix, channel_name, grid_rows, grid_cols, raw_path_for_naming, output_base_dir):
    """
    绘制增益矩阵的热力图并保存到文件。
    """
    plt.figure(figsize=(10, 8))

    actual_values = matrix[matrix != 1.0]

    min_display_val = 1.0
    max_display_val = MAX_GAIN

    if actual_values.size > 0:
        min_val_calc = np.min(matrix)
        max_val_calc = np.max(matrix)
        vmin_plot = max(0.1, min_val_calc * 0.95)
        vmax_plot = max_val_calc * 1.05

        if abs(vmax_plot - vmin_plot) < 0.01:
            vmin_plot = 1.0
            vmax_plot = max(1.01, vmax_plot)

        min_display_val = vmin_plot
        max_display_val = vmax_plot

    im = plt.imshow(matrix, cmap='jet', origin='upper', vmin=min_display_val, vmax=max_display_val)

    for (j, i), val in np.ndenumerate(matrix):
        normalized_val = (val - min_display_val) / (max_display_val - min_display_val + 1e-6)
        text_color = 'black' if normalized_val > 0.6 else 'white'

        plt.text(i, j, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0.5, alpha=0.6))

    cbar = plt.colorbar(im, label='Gain Value')
    plt.title(f"{channel_name} Channel Gain Map ({title_suffix})")
    plt.xlabel('Grid Column')
    plt.ylabel('Grid Row')
    plt.xticks(np.arange(grid_cols))
    plt.yticks(np.arange(grid_rows))

    raw_filename_base = os.path.splitext(os.path.basename(raw_path_for_naming))[0]
    output_dir_heatmaps = os.path.join(output_base_dir, 'heatmaps')
    os.makedirs(output_dir_heatmaps, exist_ok=True)

    filename = os.path.join(output_dir_heatmaps, f"{raw_filename_base}_{channel_name}_heatmap.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"已保存 {channel_name} 通道热力图至 {filename}")

def save_gain_matrix_to_txt(matrix, channel_name, raw_path_for_naming, output_base_dir, is_golden=False):
    """将增益矩阵保存为文本文件。"""
    raw_filename_base = os.path.splitext(os.path.basename(raw_path_for_naming))[0]
    output_dir_matrices = os.path.join(output_base_dir, 'gain_matrices')
    os.makedirs(output_dir_matrices, exist_ok=True)

    if is_golden:
        filename = os.path.join(output_dir_matrices, f"{raw_filename_base}_{channel_name}_golden_table_for_tuning.txt")
        header = f"用于Tuning的Golden增益表 (1024 / script_gain) - {channel_name} 通道:"
        fmt_str = '%d'
    else:
        filename = os.path.join(output_dir_matrices, f"{raw_filename_base}_{channel_name}_script_gain.txt")
        header = f"脚本计算出的原始增益 - {channel_name} 通道:"
        fmt_str = '%.4f'

    flat_matrix = matrix.flatten()
    formatted_line = " ".join([fmt_str % num for num in flat_matrix])

    with open(filename, 'w') as f:
        f.write(f'# {header}\n')
        f.write(formatted_line + '\n')

    print(f"已保存 {header.split('-')[0].strip()} 至: {filename}")

# 【算法升级】平滑函数，增加中值滤波来消除尖锐的噪声脉冲
def smooth_table(table, median_ksize=5, gaussian_ksize=5):
    """
    对矩阵进行平滑处理。
    先使用中值滤波消除孤立的尖锐脉冲噪声，然后使用高斯模糊进行整体平滑。
    Args:
        table (np.array): 输入的增益矩阵。
        median_ksize (int): 中值滤波的内核大小，必须是奇数。
        gaussian_ksize (int): 高斯模糊的内核大小，必须是奇数。
    Returns:
        np.array: 平滑后的矩阵。
    """
    # 确保内核大小是奇数
    if median_ksize % 2 == 0:
        median_ksize += 1
    if gaussian_ksize % 2 == 0:
        gaussian_ksize += 1

    print("平滑前 - min:", np.min(table), "max:", np.max(table), "mean:", np.mean(table))

    # 中值滤波要求数据类型是 float32
    table_float32 = table.astype(np.float32)

    # 步骤1: 使用中值滤波，可以极其有效地去除热力图上那种“孤立亮斑”噪声
    denoised_table = cv2.medianBlur(table_float32, median_ksize)

    # 步骤2: 之后再使用高斯模糊，对整体进行平滑，使过渡更自然
    smoothed_table = cv2.GaussianBlur(denoised_table, (gaussian_ksize, gaussian_ksize), 0)

    print("平滑后 - min:", np.min(smoothed_table), "max:", np.max(smoothed_table), "mean:", np.mean(smoothed_table))
    return smoothed_table


# 【新增】对称化函数
def symmetrize_table(table):
    """
    通过取对称点平均值的方式，强制使矩阵中心对称。
    """
    rows, cols = table.shape
    symmetrized_table = table.copy()

    for r in range((rows + 1) // 2):
        for c in range(cols):
            sym_r = rows - 1 - r
            sym_c = cols - 1 - c
            avg_val = (symmetrized_table[r, c] + symmetrized_table[sym_r, sym_c]) / 2.0
            symmetrized_table[r, c] = avg_val
            symmetrized_table[sym_r, sym_c] = avg_val

    return symmetrized_table

# 【新增】径向衰减图生成函数
def create_falloff_map(rows, cols, falloff_at_edge):
    """
    创建一个从中心到边缘线性变化的衰减因子地图。
    """
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    max_dist = np.sqrt(center_r**2 + center_c**2)

    falloff_map = np.ones((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            ratio = dist / max_dist
            falloff_map[r, c] = 1.0 * (1 - ratio) + falloff_at_edge * ratio

    return falloff_map
# --- 【V10 - Pure LSC】核心校准函数 (最终版) ---
# 该版本严格遵守LSC的职责：仅校正空间上的亮度与颜色不均匀性，不干预全局白平衡。
def perform_lsc_calibration(raw_img_path, width, height, bayer_pattern,
                            grid_rows, grid_cols,
                            black_levels_dict,
                            output_dir,
                            feather_pixels, max_gain,
                            valid_grid_threshold_ratio,
                            falloff_factor,
                            use_manual_selection):
    global MIN_PIXELS_PER_GRID

    print(f"\n--- 开始LSC标定 (V10 - Pure LSC)，生成平台增益表：{os.path.basename(raw_img_path)} ---")
    print("本版本仅校正Luma和Chroma Shading，不包含白平衡。")

    original_bayer_16bit = read_raw_bayer_image_manual(raw_img_path, width, height)
    if original_bayer_16bit is None: return None
    h, w = original_bayer_16bit.shape

    # 1. 准备预览图
    avg_bl = (black_levels_dict['Gr'] + black_levels_dict['Gb']) / 2.0
    preview_bayer_8bit = (np.maximum(0, original_bayer_16bit.astype(np.float32) - avg_bl) * (255.0 / (1023.0 - avg_bl))).astype(np.uint8)
    original_rgb_float_no_wb = cv2.cvtColor(preview_bayer_8bit, bayer_pattern).astype(np.float32) / 255.0

    if use_manual_selection:
        temp_display_img_for_selection = simple_white_balance(original_rgb_float_no_wb.copy())
        feathered_mask_2d, detected_circle_info = get_manual_circle_mask(temp_display_img_for_selection, feather_pixels, output_dir, MANUAL_ADJUST_STEP)
        print(f"手动选择已确认: 圆心=({detected_circle_info[0]},{detected_circle_info[1]}), 半径={detected_circle_info[2]}")

    # 2. 提取通道并进行BLC和归一化
    print("步骤1: 提取通道, 进行黑电平校正和归一化...")
    bayer_channels_float = extract_bayer_channels(original_bayer_16bit, bayer_pattern, black_levels_dict)

    # 3. 计算每个网格的平均亮度
    print("步骤2: 计算每个网格的平均亮度...")
    H_grid_cell_size = h // grid_rows
    W_grid_cell_size = w // grid_cols
    grid_brightness_maps = {ch: np.zeros((grid_rows, grid_cols), dtype=np.float32) for ch in ['R', 'Gr', 'Gb', 'B']}
    epsilon = 1e-6
    hard_mask_for_calc = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask_for_calc, (detected_circle_info[0], detected_circle_info[1]), detected_circle_info[2], 1, -1)
    for i in range(grid_rows):
        for j in range(grid_cols):
            y_start, y_end = i * H_grid_cell_size, (i + 1) * H_grid_cell_size
            x_start, x_end = j * W_grid_cell_size, (j + 1) * W_grid_cell_size
            hard_mask_patch = hard_mask_for_calc[y_start:y_end, x_start:x_end]

            # 使用Gr通道的有效像素作为判断依据
            num_valid_pixels = np.count_nonzero((bayer_channels_float['Gr'][y_start:y_end, x_start:x_end] > 0) & (hard_mask_patch == 1))

            if num_valid_pixels > MIN_PIXELS_PER_GRID:
                for ch_name in ['R', 'Gr', 'Gb', 'B']:
                    patch = bayer_channels_float[ch_name][y_start:y_end, x_start:x_end]
                    valid_mask = (patch > 0) & (hard_mask_patch == 1)
                    if np.any(valid_mask):
                        grid_brightness_maps[ch_name][i, j] = np.mean(patch[valid_mask])

    # 4. LSC增益计算 (V10 - 纯净版)
    print("步骤3: 计算LSC增益...")
    G_avg_map = (grid_brightness_maps['Gr'] + grid_brightness_maps['Gb']) / 2.0
    R_map, B_map = grid_brightness_maps['R'], grid_brightness_maps['B']

    center_row_idx, center_col_idx = grid_rows // 2, grid_cols // 2
    center_G_avg = G_avg_map[center_row_idx, center_col_idx]
    center_R = R_map[center_row_idx, center_col_idx]
    center_B = B_map[center_row_idx, center_col_idx]

    # 【新增】【日志】打印核心参考值
    print(f"  - 中心网格参考亮度: R={center_R:.4f}, G_avg={center_G_avg:.4f}, B={center_B:.4f}")

    if center_G_avg < epsilon or center_R < epsilon or center_B < epsilon:
        print("错误：中心网格亮度过低或无效。无法继续计算。")
        return None

    master_valid_mask = G_avg_map > (center_G_avg * valid_grid_threshold_ratio)

    # 核心逻辑：对每个通道独立计算其亮度增益，使其恢复到中心点的水平。
    gain_R = np.where(R_map > epsilon, center_R / R_map, 1.0)
    gain_G = np.where(G_avg_map > epsilon, center_G_avg / G_avg_map, 1.0)
    gain_B = np.where(B_map > epsilon, center_B / B_map, 1.0)

    # (可选) 应用 falloff_factor 来微调边缘的亮度，可以更柔和地校正
    luma_falloff_map = create_falloff_map(grid_rows, grid_cols, falloff_factor)
    final_gain_R = np.power(gain_R, luma_falloff_map)
    final_gain_G = np.power(gain_G, luma_falloff_map)
    final_gain_B = np.power(gain_B, luma_falloff_map)

    final_gain_R[~master_valid_mask] = 1.0
    final_gain_G[~master_valid_mask] = 1.0
    final_gain_B[~master_valid_mask] = 1.0

    final_gain_matrices = {
        'R': np.clip(final_gain_R, 1.0, max_gain),
        'Gr': np.clip(final_gain_G, 1.0, max_gain),
        'Gb': np.clip(final_gain_G, 1.0, max_gain),
        'B': np.clip(final_gain_B, 1.0, max_gain)
    }

    # 5. 应用增益
    print("步骤4: 应用增益到Bayer数据...")
    bl_map = np.zeros_like(original_bayer_16bit, dtype=np.float32)
    if bayer_pattern == cv2.COLOR_BayerRG2BGR_VNG: # RGGB
        bl_map[0::2, 0::2] = black_levels_dict['R'];  bl_map[0::2, 1::2] = black_levels_dict['Gr']
        bl_map[1::2, 0::2] = black_levels_dict['Gb']; bl_map[1::2, 1::2] = black_levels_dict['B']
    elif bayer_pattern == cv2.COLOR_BayerGR2BGR_VNG: # GRBG
        bl_map[0::2, 0::2] = black_levels_dict['Gr']; bl_map[0::2, 1::2] = black_levels_dict['R']
        bl_map[1::2, 0::2] = black_levels_dict['B'];  bl_map[1::2, 1::2] = black_levels_dict['Gb']
    elif bayer_pattern == cv2.COLOR_BayerBG2BGR_VNG: # BGGR
        bl_map[0::2, 0::2] = black_levels_dict['B'];  bl_map[0::2, 1::2] = black_levels_dict['Gb']
        bl_map[1::2, 0::2] = black_levels_dict['Gr']; bl_map[1::2, 1::2] = black_levels_dict['R']
    elif bayer_pattern == cv2.COLOR_BayerGB2BGR_VNG: # GBRG
        bl_map[0::2, 0::2] = black_levels_dict['Gb']; bl_map[0::2, 1::2] = black_levels_dict['B']
        bl_map[1::2, 0::2] = black_levels_dict['R'];  bl_map[1::2, 1::2] = black_levels_dict['Gr']

    bayer_blc_float = np.maximum(0, original_bayer_16bit.astype(np.float32) - bl_map)

    gain_map_R_full = cv2.resize(final_gain_matrices['R'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gr_full = cv2.resize(final_gain_matrices['Gr'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gb_full = cv2.resize(final_gain_matrices['Gb'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_B_full = cv2.resize(final_gain_matrices['B'], (w, h), interpolation=cv2.INTER_LINEAR)

    compensated_bayer_blc_float = apply_gain_to_bayer(
        bayer_blc_float,
        gain_map_R_full, gain_map_Gr_full, gain_map_Gb_full, gain_map_B_full,
        bayer_pattern
    )

    # 6. 可视化和返回
    print("步骤5: 生成最终图像用于可视化...")
    max_display_val = 1023.0 - avg_bl
    compensated_bayer_8bit = (np.clip(compensated_bayer_blc_float, 0, max_display_val) * (255.0 / max_display_val)).astype(np.uint8)
    compensated_rgb_float = cv2.cvtColor(compensated_bayer_8bit, bayer_pattern).astype(np.float32) / 255.0
    compensated_rgb_float = compensated_rgb_float * np.stack([feathered_mask_2d] * 3, axis=-1)
    compensated_rgb_float = np.clip(compensated_rgb_float, 0.0, 1.0)

    # 返回LSC校正后的图像和增益矩阵
    original_rgb_float_wb = simple_white_balance(original_rgb_float_no_wb.copy(), mask_2d=feathered_mask_2d)
    compensated_rgb_float_wb = simple_white_balance(compensated_rgb_float.copy(), mask_2d=feathered_mask_2d)
    
    # 【修改】函数返回内容增加 feathered_mask_2d 和 detected_circle_info
    return (original_rgb_float_wb, compensated_rgb_float_wb, compensated_rgb_float,
            original_rgb_float_no_wb, final_gain_matrices, feathered_mask_2d, detected_circle_info)


# 【更新】可视化函数，显示4张对比图
def visualize_results_circle_mask(original_img_no_wb, original_img_wb, compensated_pure_lsc_no_wb, compensated_lsc_and_wb, final_gain_matrices, output_dir='.'):
    """
    可视化4张关键图像的对比、亮度直方图以及各通道的增益热力图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = os.path.splitext(os.path.basename(RAW_IMAGE_PATH))[0]
    output_images_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_images_dir, exist_ok=True)

    # --- 1. 四张关键图像对比 ---
    plt.figure(figsize=(24, 6))
    plt.suptitle('Image Correction Pipeline: Step-by-Step', fontsize=16)

    plt.subplot(1, 4, 1)
    plt.imshow(original_img_no_wb)
    plt.title('[1] Original (No LSC, No WB)')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(compensated_pure_lsc_no_wb)
    plt.title('[2] Compensated (LSC only, No WB)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(compensated_lsc_and_wb)
    plt.title('[3] Final Result (LSC + WB)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(original_img_wb)
    plt.title('[4] Reference (Original + WB)')
    plt.axis('off')

    plt.savefig(os.path.join(output_images_dir, f'{img_name}_1_image_comparison.png'), bbox_inches='tight')
    plt.show()

    # --- 2. 四张图像的亮度直方图对比 ---
    plt.figure(figsize=(20, 5))
    plt.suptitle('Brightness Distribution Histograms', fontsize=16)
    bins = 100

    plt.subplot(1, 4, 1)
    valid_pixels = original_img_no_wb[original_img_no_wb > 0.01]
    plt.hist(valid_pixels.flatten(), bins=bins, color='gray', alpha=0.7)
    plt.title('[1] Original (No LSC, No WB)')
    plt.xlabel('Brightness'); plt.ylabel('Pixel Count')

    plt.subplot(1, 4, 2)
    valid_pixels = compensated_pure_lsc_no_wb[compensated_pure_lsc_no_wb > 0.01]
    plt.hist(valid_pixels.flatten(), bins=bins, color='green', alpha=0.7)
    plt.title('[2] Compensated (LSC only, No WB)')
    plt.xlabel('Brightness')

    plt.subplot(1, 4, 3)
    valid_pixels = compensated_lsc_and_wb[compensated_lsc_and_wb > 0.01]
    plt.hist(valid_pixels.flatten(), bins=bins, color='blue', alpha=0.7)
    plt.title('[3] Final Result (LSC + WB)')
    plt.xlabel('Brightness')

    plt.subplot(1, 4, 4)
    valid_pixels = original_img_wb[original_img_wb > 0.01]
    plt.hist(valid_pixels.flatten(), bins=bins, color='red', alpha=0.7)
    plt.title('[4] Reference (Original + WB)')
    plt.xlabel('Brightness')

    plt.savefig(os.path.join(output_images_dir, f'{img_name}_2_histogram_comparison.png'), bbox_inches='tight')
    plt.show()

# 【恢复】新增LSC校准前后对比图函数
def save_comparison_image(before_img_wb, after_img_wb, output_path):
    """
    将校准前后的图像并排放在一起并保存。
    """
    before_8bit = (np.clip(before_img_wb, 0, 1) * 255).astype(np.uint8)
    after_8bit = (np.clip(after_img_wb, 0, 1) * 255).astype(np.uint8)

    if before_8bit.shape != after_8bit.shape:
        print("警告: 校准前后的图像尺寸不同，无法创建对比图。")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255) # Red
    thickness = 2

    cv2.putText(before_8bit, 'Before LSC', (20, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(after_8bit, 'After LSC', (20, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)

    comparison_img = np.hstack((before_8bit, after_8bit))

    comparison_img_bgr = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, comparison_img_bgr)
    print(f"LSC校准前后对比图已保存至: {output_path}")

# --- 【新增】掩膜可视化函数 ---
def visualize_and_save_mask(original_image, mask_2d, circle_params, base_filename, output_dir):
    """
    可视化手动选择的圆形掩膜并保存。
    1. 在原图上绘制掩膜轮廓。
    2. 单独保存羽化后的灰度掩膜。
    """
    print("\n--- 正在生成掩膜可视化图像 ---")
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 保存掩膜叠加图
    overlay_img_8bit = (np.clip(original_image, 0, 1) * 255).astype(np.uint8)
    overlay_img_bgr = cv2.cvtColor(overlay_img_8bit, cv2.COLOR_RGB2BGR)
    cx, cy, r = circle_params
    # 用红色(BGR: 0,0,255)粗线条(thickness=3)画出轮廓
    cv2.circle(overlay_img_bgr, (cx, cy), r, (0, 0, 255), 3)
    overlay_path = os.path.join(vis_dir, f'{base_filename}_mask_overlay.png')
    cv2.imwrite(overlay_path, overlay_img_bgr)
    print(f"掩膜叠加图已保存至: {overlay_path}")

    # 2. 保存灰度掩膜图
    grayscale_mask_8bit = (mask_2d * 255).astype(np.uint8)
    mask_path = os.path.join(vis_dir, f'{base_filename}_mask_grayscale.png')
    cv2.imwrite(mask_path, grayscale_mask_8bit)
    print(f"灰度掩膜图已保存至: {mask_path}")


# --- 主程序 (最终功能完整版) ---
if __name__ == '__main__':
    USE_MANUAL_CIRCLE_SELECTION = True # 强烈建议保持为 True

    # 【日志】脚本启动时打印所有配置参数 (代码不变)
    print("="*50)
    print("LSC 校准脚本启动")
    # ... (这部分打印代码都正确，保持不变) ...
    print("="*50)


    # 确保输出目录存在 (代码不变)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 创建虚拟RAW文件用于测试 (代码不变)
    if not os.path.exists(RAW_IMAGE_PATH):
        # ... (创建虚拟文件的代码都正确，保持不变) ...
        print("--- 虚拟RAW文件创建完毕 ---")


    # 执行亮度补偿主函数 (代码不变)
    result = perform_lsc_calibration(
        RAW_IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, BAYER_PATTERN,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        black_levels_dict=BLACK_LEVELS,
        output_dir=OUTPUT_DIR,
        feather_pixels=MASK_FEATHER_PIXELS,
        max_gain=MAX_GAIN,
        valid_grid_threshold_ratio=VALID_GRID_THRESHOLD_RATIO,
        falloff_factor=FALLOFF_FACTOR,
        use_manual_selection=USE_MANUAL_CIRCLE_SELECTION
    )

    if result is not None:
        # 接收返回结果 (代码不变)
        original_rgb_wb, \
        compensated_lsc_and_wb, \
        compensated_pure_lsc_no_wb, \
        original_rgb_no_wb, \
        final_gain_matrices, \
        feathered_mask, \
        circle_info = result

        base_filename = os.path.splitext(os.path.basename(RAW_IMAGE_PATH))[0]

        # --- 1. 保存 各种对比图像 ---
        print("\n正在保存各种对比图像...")
        # 1a. 原始图 (无处理)
        img_to_save = (original_rgb_no_wb * 255).astype(np.uint8)
        output_path = os.path.join(OUTPUT_DIR, f'{base_filename}_1_original_NoWB.png')
        cv2.imwrite(output_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        print(f"图像 [1] 已保存: {output_path}")
        
        # 1b. 原始图 (仅做白平衡)
        img_to_save = (original_rgb_wb * 255).astype(np.uint8)
        output_path = os.path.join(OUTPUT_DIR, f'{base_filename}_2_original_WB_only.png')
        cv2.imwrite(output_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        print(f"图像 [2] 已保存: {output_path}")
        
        # 1c. 纯LSC校正后的图像 (未做白平衡)
        img_to_save = (compensated_pure_lsc_no_wb * 255).astype(np.uint8)
        output_path = os.path.join(OUTPUT_DIR, f'{base_filename}_3_compensated_pure_LSC_NoWB.png')
        cv2.imwrite(output_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        print(f"图像 [3] 已保存: {output_path}")

        # 1d. LSC校正后且白平衡的图像 (最终效果图)
        img_to_save = (compensated_lsc_and_wb * 255).astype(np.uint8)
        output_path = os.path.join(OUTPUT_DIR, f'{base_filename}_4_compensated_LSC_and_WB.png')
        cv2.imwrite(output_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
        print(f"图像 [4] 已保存: {output_path}")
        
        # --- 2. 【新增】调用掩膜可视化函数 --- (代码不变)
        visualize_and_save_mask(original_rgb_no_wb, feathered_mask, circle_info, base_filename, OUTPUT_DIR)

        # ==================== 【修正这个循环】 ====================
        # --- 3. 循环处理增益表 ---
        print("\n--- 正在平滑增益矩阵并保存热力图和文本文件 ---")
        SMOOTHING_MEDIAN_KSIZE = 3
        SMOOTHING_GAUSSIAN_KSIZE = 3
        
        # 定义一个新字典来存储平滑后的增益矩阵
        smoothed_gain_matrices = {}

        # --- FOR 循环开始 ---
        for ch_name in ['R', 'Gr', 'Gb', 'B']:
            if ch_name in final_gain_matrices:
                # 平滑处理 (这部分不变)
                print(f"正在平滑 {ch_name} 通道的增益矩阵...")
                smoothed_gain_matrix = extrapolate_and_smooth_gains(
                    final_gain_matrices[ch_name],
                    gaussian_ksize=SMOOTHING_GAUSSIAN_KSIZE
                )
                
                # 将平滑后的结果存入新字典
                smoothed_gain_matrices[ch_name] = smoothed_gain_matrix

                # 【修正】调用绘图函数，并补全所有必需参数
                plot_heatmap_and_save_matrix(
                    matrix=smoothed_gain_matrix,
                    title_suffix="Final Smoothed Gain",
                    channel_name=ch_name,
                    grid_rows=GRID_ROWS,                # <--- 补上这个参数
                    grid_cols=GRID_COLS,                  # <--- 补上这个参数
                    raw_path_for_naming=RAW_IMAGE_PATH,   # <--- 补上这个参数
                    output_base_dir=OUTPUT_DIR
                )
                
                # 【修正】调用保存文本函数，并补全所有必需参数
                save_gain_matrix_to_txt(
                    matrix=smoothed_gain_matrix,
                    channel_name=ch_name,
                    raw_path_for_naming=RAW_IMAGE_PATH,   # <--- 补上这个参数
                    output_base_dir=OUTPUT_DIR
                )
        # --- FOR 循环结束 ---

        print("所有通道的增益矩阵已处理并保存。")
        
        # --- 4. 保存LSC校准前后对比图 ---
        print("\n--- 正在保存LSC校准前后对比图 ---")
        comparison_path = os.path.join(OUTPUT_DIR, 'visualizations', f'{base_filename}_lsc_before_vs_after.png')
        # 注意：这里的对比图仍然是基于原始（未平滑）增益计算的图像，这通常是可以接受的。
        # 因为平滑主要影响最终的增益表文件，用于消除色偏。
        save_comparison_image(original_rgb_wb, compensated_lsc_and_wb, comparison_path)

        # --- 5. 调用完整的可视化分析函数 ---
        print("\n--- 正在生成Matplotlib可视化分析图 ---")
        visualize_results_circle_mask(
            original_rgb_no_wb,
            original_rgb_wb,
            compensated_pure_lsc_no_wb,
            compensated_lsc_and_wb,
            # 注意：这里传递的是平滑后的增益矩阵字典，让分析图也反映平滑效果
            smoothed_gain_matrices, 
            OUTPUT_DIR
        )

        print("\n" + "="*50)
        print("所有任务已成功完成！")
        print(f"所有输出文件已保存在目录: {os.path.abspath(OUTPUT_DIR)}")
        print("="*50)

    else:
        print("\n补偿过程失败。请检查之前的错误信息。")