# ===================================================================================
# LSC (Lens Shading Correction) Calibration Script
# Version: V15.5 - Fixed Center Point
#
# Description:
# This script is designed to generate LSC gain tables for fisheye lenses,
# specifically tailored for platforms like Qualcomm's. It addresses issues
# such as luma shading, chroma shading, and bright halos at the edges for
# panoramic stitching.
#
# Core Algorithm:
# "G-Channel Priority with Color Ratio Lock" combined with a comprehensive
# robustness pipeline.
#
# Changes in this Version:
# - Reverted to using a fixed geometric center point for calibration as per user feedback,
#   as the dynamic center point caused issues in some cases.
# - Kept the "Black Level Correction" and "Remember Chosen Region" features.
# ===================================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# --- 配置参数 ---
# 请根据你的实际文件和相机参数进行修改！
RAW_IMAGE_PATH = '0715_25k.raw' # 假设你的裸RAW文件路径
IMAGE_WIDTH = 1256 # 你的RAW图像宽度
IMAGE_HEIGHT = 1256 # 你的RAW图像高度

# 【功能】黑电平参数 (根据你的传感器规格设定，通常是10-bit或12-bit数据)
# 例如，对于10-bit RAW (0-1023)，黑电平可能是64。
BLACK_LEVELS = {'R': 64, 'Gr': 64, 'Gb': 64, 'B': 64}

# 根据你的相机传感器实际拜耳模式选择。常见的有：
# cv2.COLOR_BayerBG2BGR_VNG  # BGGR 模式
# cv2.COLOR_BayerGR2BGR_VNG  # GRBG 模式
# cv2.COLOR_BayerRG2BGR_VNG  # RGGB 模式 (请根据实际情况修改)
# cv2.COLOR_BayerGB2BGR_VNG  # GBRG 模式
BAYER_PATTERN =  cv2.COLOR_BayerGR2BGR_VNG # 默认为 GRBG 模式，请务必根据实际情况修改！
OUTPUT_DIR = 'output_images_manual_raw_interactive' # 修改输出目录，便于区分

# 此时应关闭此选项，让算法根据真实数据进行非对称校正。
APPLY_SYMMETRY = False # 设为 False 来解决左右色偏问题
# 【关键参数】衰减因子 (Falloff Factor)。
# 该值控制LSC校正在【边界处】的补偿强度。小于1.0会减弱对边缘暗角的补偿，从而避免产生亮环。
# 建议值范围：0.7 - 1.0。
FALLOFF_FACTOR = 1
# 亮度补偿网格数量
GRID_ROWS = 13
GRID_COLS = 17
# 掩码羽化
MASK_FEATHER_PIXELS = 120
# 增益裁剪限制
MAX_GAIN = 2.5
# 【新增关键参数】颜色校正衰减因子
COLOR_FALLOFF_FACTOR = 1

# 【关键参数】网格有效性门槛比例。
# 如果一个网格的平均G通道亮度低于 (中心G通道亮度 * 此比例)，则该网格被视为无效，其所有增益将被设为1.0。
# 这可以有效防止对鱼眼镜头边缘的过暗区域进行过度补偿。建议值范围：0.02 - 0.1 (即2%到10%)
VALID_GRID_THRESHOLD_RATIO = 0.2

# --- 新增参数：手动调整的步长 ---
MANUAL_ADJUST_STEP = 1 # 调整步长，可以设置为1、2、5等，数值越小精度越高
# --- Matplotlib 中文字体配置 (此函数将不再配置中文字体，而是确保英文显示正常) ---
def set_matplotlib_english_font():
    # 确保 Matplotlib 使用非中文字体，以避免混淆
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] # 常用英文字体，作为回退
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
    print("Matplotlib has been configured for English display.")
# 在脚本开始时调用字体配置函数
set_matplotlib_english_font()
# --- 辅助函数：手动读取和处理裸RAW数据 (修改为直接返回16位拜耳数据) ---
def read_raw_bayer_image_manual(raw_path, width, height):
    """
    手动读取10-bit或16-bit的裸RAW数据，并返回16-bit的Numpy数组。
    Args:
        raw_path (str): 裸RAW文件路径。
        width (int): 图像宽度。
        height (int): 图像高度。
    Returns:
        np.array: 16-bit的拜耳原始图像数据。
    """
    try:
        expected_pixels = width * height
        # 核心修改：明确指定读取的元素数量为 expected_pixels
        bayer_data_16bit_raw = np.fromfile(raw_path, dtype='<u2', count=expected_pixels) # '<u2' for little-endian uint16

        # 再次检查读取到的元素数量是否与预期相符，以防文件过小
        if bayer_data_16bit_raw.size != expected_pixels:
            raise ValueError(f"Raw file size mismatch after attempting to read expected count. Expected {expected_pixels} pixels, but got {bayer_data_16bit_raw.size}.")

        bayer_image_16bit = bayer_data_16bit_raw.reshape((height, width))
        return bayer_image_16bit
    except Exception as e:
        print(f"Error reading raw image: {e}")
        return None

def extract_bayer_channels(bayer_blc_float, bayer_pattern_code):
    """
    【V3】将已经去除黑电平的浮点bayer数据分离并归一化。
    Args:
        bayer_blc_float (np.array): 已经减去黑电平的浮点bayer图像数据。
        bayer_pattern_code (int): OpenCV的拜耳模式常量。
    Returns:
        dict: 包含 'R', 'Gr', 'Gb', 'B' 键的字典。
    """
    h, w = bayer_blc_float.shape
    # 1. 归一化。假设信号范围是 1023 (10-bit)。
    # 注意：此时黑电平已经是0，所以直接除以最大值即可。
    normalized_bayer = bayer_blc_float / (1023.0 - 64.0) # 使用平均G通道黑电平作为基准
    normalized_bayer = np.clip(normalized_bayer, 0, 1.0)

    # 2. 初始化四个独立的通道
    R, Gr, Gb, B = [np.zeros((h, w), dtype=np.float32) for _ in range(4)]

    # 3. 根据拜耳模式，将归一化后的数据分离到对应的通道矩阵中
    if bayer_pattern_code == cv2.COLOR_BayerRG2BGR_VNG: # RGGB
        R[0::2, 0::2] = normalized_bayer[0::2, 0::2]
        Gr[0::2, 1::2] = normalized_bayer[0::2, 1::2]
        Gb[1::2, 0::2] = normalized_bayer[1::2, 0::2]
        B[1::2, 1::2] = normalized_bayer[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGR2BGR_VNG: # GRBG
        Gr[0::2, 0::2] = normalized_bayer[0::2, 0::2]
        R[0::2, 1::2] = normalized_bayer[0::2, 1::2]
        B[1::2, 0::2] = normalized_bayer[1::2, 0::2]
        Gb[1::2, 1::2] = normalized_bayer[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerBG2BGR_VNG: # BGGR
        B[0::2, 0::2] = normalized_bayer[0::2, 0::2]
        Gb[0::2, 1::2] = normalized_bayer[0::2, 1::2]
        Gr[1::2, 0::2] = normalized_bayer[1::2, 0::2]
        R[1::2, 1::2] = normalized_bayer[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGB2BGR_VNG: # GBRG
        Gb[0::2, 0::2] = normalized_bayer[0::2, 0::2]
        B[0::2, 1::2] = normalized_bayer[0::2, 1::2]
        R[1::2, 0::2] = normalized_bayer[1::2, 0::2]
        Gr[1::2, 1::2] = normalized_bayer[1::2, 1::2]
    else:
        raise ValueError("Unsupported Bayer pattern code.")

    return {'R': R, 'Gr': Gr, 'Gb': Gb, 'B': B}

# --- 【V3 - 逻辑重构】---
def apply_gain_to_bayer(bayer_blc_float, gain_map_R, gain_map_Gr, gain_map_Gb, gain_map_B, bayer_pattern_code):
    """
    【V3】将增益图应用到已经去除黑电平的浮点bayer数据上。
    """
    compensated_bayer_float = bayer_blc_float.copy()

    if bayer_pattern_code == cv2.COLOR_BayerRG2BGR_VNG: # RGGB
        compensated_bayer_float[0::2, 0::2] *= gain_map_R[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_Gr[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_Gb[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_B[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGR2BGR_VNG: # GRBG
        compensated_bayer_float[0::2, 0::2] *= gain_map_Gr[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_R[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_B[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_Gb[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerBG2BGR_VNG: # BGGR
        compensated_bayer_float[0::2, 0::2] *= gain_map_B[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_Gb[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_Gr[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_R[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGB2BGR_VNG: # GBRG
        compensated_bayer_float[0::2, 0::2] *= gain_map_Gb[0::2, 0::2]
        compensated_bayer_float[0::2, 1::2] *= gain_map_B[0::2, 1::2]
        compensated_bayer_float[1::2, 0::2] *= gain_map_R[1::2, 0::2]
        compensated_bayer_float[1::2, 1::2] *= gain_map_Gr[1::2, 1::2]

    return compensated_bayer_float # 返回的也是浮点数

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
            print("Warning: Green channel mean in the central masked area is too low. Skipping white balance.")
            return image_rgb_float

        avg_R = np.mean(central_patch_R[valid_pixels_mask])
        avg_G = np.mean(g_channel_valid)
        avg_B = np.mean(central_patch_B[valid_pixels_mask])

    else:
        avg_G = np.mean(central_patch_G)
        if avg_G < 1e-6:
            print("Warning: Average Green channel in the central area is too low. Skipping white balance.")
            return image_rgb_float
        avg_R = np.mean(central_patch_R)
        avg_B = np.mean(central_patch_B)

    if avg_G < 1e-6:
        print("Warning: Average Green channel is too low for white balance. Skipping white balance.")
        return image_rgb_float

    gain_R = avg_G / (avg_R + 1e-6)
    gain_B = avg_G / (avg_B + 1e-6)

    print(f"Robust White Balance Gains (from central patch): R={gain_R:.2f}, G=1.00, B={gain_B:.2f}")

    balanced_image[:, :, 0] = np.clip(R * gain_R, 0, 1.0)
    balanced_image[:, :, 2] = np.clip(B * gain_B, 0, 1.0)

    return balanced_image

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
        print(f"\n--- Found previous selection in '{params_path}' ---")
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
            
            print("Options:")
            print(" 'r': Reuse the previous selection.")
            print(" 'e': Edit the previous selection.")
            print(" 'n' or any other key: Start a new selection.")
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                print("Reusing previous selection.")
                current_circle = preview_circle
                start_main_loop = False # 直接跳过手动调整循环
            elif key == ord('e'):
                print("Editing previous selection.")
                current_circle = preview_circle # 以加载的圆为起点开始调整
            else:
                print("Starting a new selection.")
        except Exception as e:
            print(f"Could not load or parse previous selection file. Error: {e}. Starting new selection.")

    if start_main_loop:
        print("\n--- Manual Circular Region Selection ---")
        print("1. Click and drag the mouse to draw a circle (from center to edge).")
        print("2. Release the mouse button to confirm the initial circle.")
        print("3. While the window is active, use the keyboard for fine-tuning:")
        print(f"   'w' / 's': Increase / Decrease Radius (Step: {adjust_step} pixels)")
        print(f"   'a' / 'd': Move Left / Right Center (Step: {adjust_step} pixels)")
        print(f"   'z' / 'x': Move Up / Down Center (Step: {adjust_step} pixels)")
        print("   'r': Reset (start drawing a new circle)")
        print("   'q': Confirm current circle and continue")
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
        print(f"Saved current circle selection to {params_path}")
    except Exception as e:
        print(f"Warning: Could not save circle parameters. Error: {e}")

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
def smooth_table(table, median_ksize=3, gaussian_ksize=3):
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
    
    # 中值滤波要求数据类型是 float32
    table_float32 = table.astype(np.float32)
    
    # 步骤1: 使用中值滤波，可以极其有效地去除热力图上那种“孤立亮斑”噪声
    denoised_table = cv2.medianBlur(table_float32, median_ksize)
    
    # 步骤2: 之后再使用高斯模糊，对整体进行平滑，使过渡更自然
    smoothed_table = cv2.GaussianBlur(denoised_table, (gaussian_ksize, gaussian_ksize), 0)
    
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

# --- 【算法升级】核心校准函数，使用固定中心点 ---
# --- 【V3 - 逻辑重构】核心校准函数 ---
def perform_lsc_calibration(raw_img_path, width, height, bayer_pattern,
                            grid_rows, grid_cols,
                            black_levels_dict,
                            output_dir,
                            feather_pixels=100, max_gain=4.0,
                            valid_grid_threshold_ratio=0.2,
                            falloff_factor=1.0, # Luma falloff
                            color_falloff_factor=1.0, # Chroma falloff
                            use_manual_selection=True):
    print(f"--- 开始LSC标定，生成平台增益表：{os.path.basename(raw_img_path)} ---")
    print(f"图像尺寸: {width}x{height}, 网格大小: {grid_rows}x{grid_cols}, 黑电平: {black_levels_dict}")
    
    original_bayer_16bit = read_raw_bayer_image_manual(raw_img_path, width, height)
    if original_bayer_16bit is None: return None, None, None, None
    h, w = original_bayer_16bit.shape

    # 1. 【核心修改】首先，创建全局的、去除黑电平后的bayer数据
    bl_map = np.zeros_like(original_bayer_16bit, dtype=np.float32)
    if bayer_pattern == cv2.COLOR_BayerGR2BGR_VNG: # GRBG
        bl_map[0::2, 0::2] = black_levels_dict['Gr']
        bl_map[0::2, 1::2] = black_levels_dict['R']
        bl_map[1::2, 0::2] = black_levels_dict['B']
        bl_map[1::2, 1::2] = black_levels_dict['Gb']
    # ... 其他bayer pattern的逻辑 ...
    else: raise ValueError(f"BLC Map creation not implemented for this Bayer Pattern: {bayer_pattern}")
    
    # 得到“纯净信号”的浮点bayer图
    bayer_blc_float = np.maximum(0, original_bayer_16bit.astype(np.float32) - bl_map)

    # 2. 准备预览图 (此部分逻辑不变)
    avg_bl = (black_levels_dict['Gr'] + black_levels_dict['Gb']) / 2.0
    preview_bayer_8bit = (np.maximum(0, original_bayer_16bit.astype(np.float32) - avg_bl) * (255.0 / (1023.0 - avg_bl))).astype(np.uint8)
    original_rgb_float_no_wb = cv2.cvtColor(preview_bayer_8bit, bayer_pattern).astype(np.float32) / 255.0
    if use_manual_selection:
        temp_display_img_for_selection = simple_white_balance(original_rgb_float_no_wb.copy())
        feathered_mask_2d, detected_circle_info = get_manual_circle_mask(temp_display_img_for_selection, feather_pixels, output_dir, MANUAL_ADJUST_STEP)
        print(f"手动选择已确认: 圆心=({detected_circle_info[0]},{detected_circle_info[1]}), 半径={detected_circle_info[2]}")
    else: return None, None, None, None

    # 3. 计算增益 (输入的是已经去除黑电平的数据)
    bayer_channels_float = extract_bayer_channels(bayer_blc_float, bayer_pattern)
    
    # ... (从这里开始到 final_gain_matrices 计算结束的所有代码，都使用您上一轮已经修改好的、包含“终极修正”的版本) ...
    H_grid_cell_size = h // grid_rows
    W_grid_cell_size = w // grid_cols
    H_grid_cell_size = max(H_grid_cell_size, 1)
    W_grid_cell_size = max(W_grid_cell_size, 1)
    grid_brightness_maps = {ch: np.zeros((grid_rows, grid_cols), dtype=np.float32) for ch in ['R', 'Gr', 'Gb', 'B']}
    epsilon = 1e-6
    for ch_name, channel_data_sparse in bayer_channels_float.items():
        for i in range(grid_rows):
            for j in range(grid_cols):
                y_start, y_end = i * H_grid_cell_size, (i + 1) * H_grid_cell_size
                x_start, x_end = j * W_grid_cell_size, (j + 1) * W_grid_cell_size
                grid_area_channel = channel_data_sparse[y_start:y_end, x_start:x_end]
                mask_area = feathered_mask_2d[y_start:y_end, x_start:x_end]
                valid_pixels = (grid_area_channel > epsilon) & (mask_area > epsilon)
                if np.any(valid_pixels):
                    grid_brightness_maps[ch_name][i, j] = np.sum(grid_area_channel[valid_pixels] * mask_area[valid_pixels]) / (np.sum(mask_area[valid_pixels]) + epsilon)
                else:
                    grid_brightness_maps[ch_name][i, j] = 0.0

    print("\n--- 开始使用最终精细算法计算增益 (V15.5) ---")
    G_avg_map = (grid_brightness_maps['Gr'] + grid_brightness_maps['Gb']) / 2.0
    R_map, B_map = grid_brightness_maps['R'], grid_brightness_maps['B']
    if np.max(G_avg_map) < epsilon:
        print("错误：图像完全黑暗，无法进行LSC计算。")
        return None, None, None, None
    center_row_idx = grid_rows // 2
    center_col_idx = grid_cols // 2
    print(f"使用固定的几何中心点: (row={center_row_idx}, col={center_col_idx}) 作为参考点。")
    center_G_avg = G_avg_map[center_row_idx, center_col_idx]
    center_R, center_B = R_map[center_row_idx, center_col_idx], B_map[center_row_idx, center_col_idx]
    if center_G_avg < epsilon:
        print("错误：中心区域G通道亮度过低，无法进行LSC计算。")
        return None, None, None, None
    validity_threshold = center_G_avg * valid_grid_threshold_ratio
    master_valid_mask = G_avg_map > validity_threshold
    print(f"Center G-luma: {center_G_avg:.4f}, Validity threshold: {validity_threshold:.4f}. Valid grids: {np.sum(master_valid_mask)}/{master_valid_mask.size}")
    target_ratio_R_G = center_R / (center_G_avg + epsilon)
    target_ratio_B_G = center_B / (center_G_avg + epsilon)
    current_ratio_R_G = np.where(master_valid_mask, R_map / (G_avg_map + epsilon), 1.0)
    ratio_correction_R = np.where(current_ratio_R_G > epsilon, target_ratio_R_G / current_ratio_R_G, 1.0)
    current_ratio_B_G = np.where(master_valid_mask, B_map / (G_avg_map + epsilon), 1.0)
    ratio_correction_B = np.where(current_ratio_B_G > epsilon, target_ratio_B_G / current_ratio_B_G, 1.0)
    ratio_correction_R_processed = smooth_table(ratio_correction_R)
    ratio_correction_B_processed = smooth_table(ratio_correction_B)
    if APPLY_SYMMETRY:
        ratio_correction_R_processed = symmetrize_table(ratio_correction_R_processed)
        ratio_correction_B_processed = symmetrize_table(ratio_correction_B_processed)
    print("Normalizing smoothed color correction tables to enforce center ratio = 1.0")
    center_ratio_R = ratio_correction_R_processed[center_row_idx, center_col_idx]
    center_ratio_B = ratio_correction_B_processed[center_row_idx, center_col_idx]
    if abs(center_ratio_R) > 1e-6: ratio_correction_R_processed /= center_ratio_R
    else: ratio_correction_R_processed.fill(1.0)
    if abs(center_ratio_B) > 1e-6: ratio_correction_B_processed /= center_ratio_B
    else: ratio_correction_B_processed.fill(1.0)
    gain_G_raw = np.where(G_avg_map > epsilon, center_G_avg / G_avg_map, 1.0)
    luma_falloff_map = create_falloff_map(grid_rows, grid_cols, falloff_factor)
    gain_G_falloff = np.power(gain_G_raw, luma_falloff_map)
    gain_G_smoothed = smooth_table(gain_G_falloff)
    final_gain_G = gain_G_smoothed
    if APPLY_SYMMETRY: final_gain_G = symmetrize_table(final_gain_G)
    print(f"Creating aggressive color falloff map, edge factor: {color_falloff_factor}")
    color_falloff_map = create_falloff_map(grid_rows, grid_cols, color_falloff_factor)
    final_gain_R = final_gain_G * (1.0 + (ratio_correction_R_processed - 1.0) * color_falloff_map)
    final_gain_B = final_gain_G * (1.0 + (ratio_correction_B_processed - 1.0) * color_falloff_map)
    final_gain_R[~master_valid_mask] = 1.0
    final_gain_G[~master_valid_mask] = 1.0
    final_gain_B[~master_valid_mask] = 1.0
    final_gain_matrices = {
        'R': np.clip(final_gain_R, 1.0, max_gain),
        'Gr': np.clip(final_gain_G, 1.0, max_gain),
        'Gb': np.clip(final_gain_G, 1.0, max_gain),
        'B': np.clip(final_gain_B, 1.0, max_gain)
    }
    
    # 4. 【核心修改】将增益应用到去除黑电平后的数据上
    gain_map_R_full = cv2.resize(final_gain_matrices['R'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gr_full = cv2.resize(final_gain_matrices['Gr'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gb_full = cv2.resize(final_gain_matrices['Gb'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_B_full = cv2.resize(final_gain_matrices['B'], (w, h), interpolation=cv2.INTER_LINEAR)

    compensated_bayer_blc_float = apply_gain_to_bayer(
        bayer_blc_float, # <--- 使用去除黑电平后的数据
        gain_map_R_full, gain_map_Gr_full, gain_map_Gb_full, gain_map_B_full,
        bayer_pattern
    )
    
    # 5. 可视化和保存
    # 注意：compensated_bayer_blc_float已经是纯信号，不需要再减黑电平
    compensated_bayer_8bit = (np.clip(compensated_bayer_blc_float, 0, 1023-avg_bl) * (255.0 / (1023.0-avg_bl))).astype(np.uint8)
    compensated_rgb_float = cv2.cvtColor(compensated_bayer_8bit, bayer_pattern).astype(np.float32) / 255.0
    compensated_rgb_float = compensated_rgb_float * np.stack([feathered_mask_2d] * 3, axis=-1)
    compensated_rgb_float = np.clip(compensated_rgb_float, 0.0, 1.0)
    
    original_rgb_float_wb = simple_white_balance(original_rgb_float_no_wb.copy(), mask_2d=feathered_mask_2d)
    compensated_rgb_float_wb = simple_white_balance(compensated_rgb_float, mask_2d=feathered_mask_2d)

    return original_rgb_float_wb, compensated_rgb_float_wb, original_rgb_float_no_wb, final_gain_matrices


# 【恢复】可视化函数
def visualize_results_circle_mask(original_img_wb, compensated_img_wb, original_img_no_wb, final_gain_matrices, output_dir='.'):
    """
    可视化原始图像（白平衡后）、补偿图像（白平衡后）、直方图以及各通道的增益热力图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = os.path.splitext(os.path.basename(RAW_IMAGE_PATH))[0]
    output_images_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(output_images_dir, exist_ok=True)

    # 1. 原始图像 (未白平衡) vs 原始图像 (白平衡) vs 补偿图像 (白平衡) 对比
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img_no_wb)
    plt.title('Original Image (Demosaiced, No WB)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(original_img_wb)
    plt.title('Original Image (Demosaiced, with WB)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(compensated_img_wb)
    plt.title('Compensated Image (LSC + WB, Feathered Mask)')
    plt.axis('off')
    plt.suptitle('Image Correction: Original vs. LSC Corrected')
    plt.savefig(os.path.join(output_images_dir, f'{img_name}_brightness_compare_all.png'), bbox_inches='tight')
    plt.show()

    # 2. 亮度直方图对比
    plt.figure(figsize=(15, 5))
    bins = 100
    
    plt.subplot(1, 3, 1)
    valid_original_pixels_no_wb = original_img_no_wb[original_img_no_wb > 0]
    plt.hist(valid_original_pixels_no_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels')
    plt.title('Original Image Brightness (No WB)')
    plt.xlabel('Normalized Brightness')
    plt.ylabel('Pixel Count')
    plt.legend()

    plt.subplot(1, 3, 2)
    valid_original_pixels_wb = original_img_wb[original_img_wb > 0]
    plt.hist(valid_original_pixels_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels')
    plt.title('Original Image Brightness (with WB)')
    plt.xlabel('Normalized Brightness')
    plt.ylabel('Pixel Count')
    plt.legend()

    plt.subplot(1, 3, 3)
    valid_compensated_pixels_wb = compensated_img_wb[compensated_img_wb > 0]
    plt.hist(valid_compensated_pixels_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels')
    plt.title('Compensated Image Brightness (LSC + WB)')
    plt.xlabel('Normalized Brightness')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.suptitle('Brightness Distribution Histograms')
    plt.savefig(os.path.join(output_images_dir, f'{img_name}_brightness_histograms_all.png'), bbox_inches='tight')
    plt.show()

# 【恢复】新增LSC校准前后对比图函数
def save_comparison_image(before_img_wb, after_img_wb, output_path):
    """
    将校准前后的图像并排放在一起并保存。
    """
    before_8bit = (np.clip(before_img_wb, 0, 1) * 255).astype(np.uint8)
    after_8bit = (np.clip(after_img_wb, 0, 1) * 255).astype(np.uint8)

    if before_8bit.shape != after_8bit.shape:
        print("Warning: Before and after images have different shapes. Cannot create comparison image.")
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

# --- 主程序 ---
if __name__ == '__main__':
    USE_MANUAL_CIRCLE_SELECTION = True # 强烈建议保持为 True

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    # 确保RAW文件所在的目录存在
    if os.path.dirname(RAW_IMAGE_PATH) and not os.path.exists(os.path.dirname(RAW_IMAGE_PATH)):
        os.makedirs(os.path.dirname(RAW_IMAGE_PATH))
        print(f"已创建RAW图像目录: {os.path.dirname(RAW_IMAGE_PATH)}。请将您的RAW图像放置于此。")
    
    # 如果RAW文件不存在，创建一个虚拟文件用于测试
    if not os.path.exists(RAW_IMAGE_PATH):
        print(f"\n--- 注意：RAW文件 '{RAW_IMAGE_PATH}' 不存在。正在创建虚拟RAW文件用于测试 ---")
        dummy_data = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
        
        # 模拟亮度衰减 (中心亮，边缘暗)
        center_x, center_y = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
        Y, X = np.ogrid[:IMAGE_HEIGHT, :IMAGE_WIDTH]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # 模拟亮度衰减，10-bit数据范围0-1023
        # 确保像素值不为0，避免log(0)等问题
        brightness_falloff = 1023 - (dist_from_center / max_dist) * 900 # 边缘亮度最低可达 1023 - 900 = 123
        
        # 对不同通道模拟不同程度的衰减，尤其是B通道，使其更暗
        if BAYER_PATTERN == cv2.COLOR_BayerRG2BGR_VNG: # RGGB
            dummy_data[0::2, 0::2] = np.clip(brightness_falloff[0::2, 0::2], 50, 1023).astype(np.uint16) # R
            dummy_data[0::2, 1::2] = np.clip(brightness_falloff[0::2, 1::2] * 0.95, 50, 1023).astype(np.uint16) # Gr
            dummy_data[1::2, 0::2] = np.clip(brightness_falloff[1::2, 0::2] * 0.95, 50, 1023).astype(np.uint16) # Gb
            dummy_data[1::2, 1::2] = np.clip(brightness_falloff[1::2, 1::2] * 0.8, 50, 1023).astype(np.uint16) # B (更暗)
        else: # 其他拜耳模式简化处理，都使用R通道的衰减模拟
            dummy_data = np.clip(brightness_falloff, 50, 1023).astype(np.uint16)

        dummy_data.tofile(RAW_IMAGE_PATH)
        print(f"已创建一个虚拟RAW文件用于测试: {RAW_IMAGE_PATH}")
        print("--- 虚拟RAW文件创建完毕 ---")

    # 执行亮度补偿主函数
    result = perform_lsc_calibration(
        RAW_IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, BAYER_PATTERN,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        black_levels_dict=BLACK_LEVELS, # 传入黑电平
        output_dir=OUTPUT_DIR, # 传入输出目录
        feather_pixels=MASK_FEATHER_PIXELS,
        max_gain=MAX_GAIN,
        valid_grid_threshold_ratio=VALID_GRID_THRESHOLD_RATIO,
        falloff_factor=FALLOFF_FACTOR,
        use_manual_selection=USE_MANUAL_CIRCLE_SELECTION
    )

    if result is not None:
        original_rgb_float_wb, compensated_rgb_final_wb, original_rgb_float_no_wb, final_gain_matrices = result
        
        base_filename = os.path.splitext(os.path.basename(RAW_IMAGE_PATH))[0]
        
        # 【恢复】保存各种PNG图像
        compensated_img_8bit_wb = (compensated_rgb_final_wb * 255).astype(np.uint8)
        output_image_path_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_compensated_feathered_mask_4ch_WB.png')
        cv2.imwrite(output_image_path_wb, cv2.cvtColor(compensated_img_8bit_wb, cv2.COLOR_RGB2BGR))
        print(f"补偿后且白平衡的图像已保存至: {output_image_path_wb}")

        original_img_8bit_wb = (original_rgb_float_wb * 255).astype(np.uint8)
        original_output_path_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_original_dem_WB.png')
        cv2.imwrite(original_output_path_wb, cv2.cvtColor(original_img_8bit_wb, cv2.COLOR_RGB2BGR))
        print(f"原始去马赛克且白平衡图像已保存至: {original_output_path_wb}")

        original_img_8bit_no_wb = (original_rgb_float_no_wb * 255).astype(np.uint8)
        original_output_path_no_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_original_dem_NoWB.png')
        cv2.imwrite(original_output_path_no_wb, cv2.cvtColor(original_img_8bit_no_wb, cv2.COLOR_RGB2BGR))
        print(f"原始去马赛克但未白平衡图像已保存至: {original_output_path_no_wb}")
        
        # 【恢复】调用新增的对比图保存函数
        comparison_path = os.path.join(OUTPUT_DIR, 'visualizations', f'{base_filename}_lsc_before_vs_after.png')
        save_comparison_image(original_rgb_float_wb, compensated_rgb_final_wb, comparison_path)

        # 【恢复】调用完整的可视化函数
        visualize_results_circle_mask(original_rgb_float_wb, compensated_rgb_final_wb, original_rgb_float_no_wb, final_gain_matrices, OUTPUT_DIR)
    else:
        print("补偿过程失败。请检查之前的错误信息。")