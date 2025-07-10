import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# --- 配置参数 ---
# 请根据你的实际文件和相机参数进行修改！
RAW_IMAGE_PATH = '25k.raw' # 假设你的裸RAW文件路径
IMAGE_WIDTH = 1256 # 你的RAW图像宽度
IMAGE_HEIGHT = 1256 # 你的RAW图像高度

# 根据你的相机传感器实际拜耳模式选择。常见的有：
# cv2.COLOR_BayerBG2BGR_VNG  # BGGR 模式
# cv2.COLOR_BayerGR2BGR_VNG  # GRBG 模式
# cv2.COLOR_BayerRG2BGR_VNG  # RGGB 模式 (请根据实际情况修改)
# cv2.COLOR_BayerGB2BGR_VNG  # GBRG 模式
BAYER_PATTERN =  cv2.COLOR_BayerGR2BGR_VNG # 默认为 GRBG 模式，请务必根据实际情况修改！
OUTPUT_DIR = 'output_images_manual_raw_interactive' # 修改输出目录，便于区分
# 亮度补偿网格数量 (例如 17x17)
GRID_ROWS = 13 # 网格行数
GRID_COLS = 17 # 网格列数 

# 掩码羽化程度（像素）。增加羽化像素宽度，可以尝试 50, 80, 100, 120等，直到过渡自然
MASK_FEATHER_PIXELS = 100 

# 【关键参数】增益裁剪限制。这是控制LSC校正强度和噪声放大之间平衡的最重要参数。
# 值越高，暗角校正越彻底，但噪声也越大。对于平台Tuning，通常建议使用2.0到4.0之间的值。
MAX_GAIN = 4.0 

# 【关键参数】衰减因子 (Falloff Factor)。
# 该值控制LSC校正的强度。小于1.0会减弱对边缘暗角的补偿，从而避免产生亮环。
# 建议值范围：0.7 - 1.0。
FALLOFF_FACTOR = 0.8

# 【关键参数】网格有效性门槛比例。
# 如果一个网格的平均G通道亮度低于 (中心G通道亮度 * 此比例)，则该网格被视为无效，其所有增益将被设为1.0。
# 这可以有效防止对鱼眼镜头边缘的过暗区域进行过度补偿。建议值范围：0.02 - 0.1 (即2%到10%)
VALID_GRID_THRESHOLD_RATIO = 0.05

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
    假设10-bit数据存储在16-bit容器中 (例如，每个像素占用2字节)。
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
            raise ValueError(f"Raw file size mismatch after attempting to read expected count. Expected {expected_pixels} pixels, but got {bayer_data_16bit_raw.size}. This might indicate the file is too small or still has an unexpected structure.")

        bayer_image_16bit = bayer_data_16bit_raw.reshape((height, width))
        return bayer_image_16bit
    except Exception as e:
        print(f"Error reading raw image: {e}")
        return None
# --- 新增函数：根据拜耳模式分离R, Gr, Gb, B通道 ---
def extract_bayer_channels(bayer_image_16bit, bayer_pattern_code):
    """
    根据拜耳模式从16-bit拜耳图像中提取R, Gr, Gb, B通道。
    返回的每个通道都是原始图像大小的矩阵，非零部分包含该通道的像素值，
    其他部分为0。像素值被归一化到0-1范围 (基于10-bit RAW数据的最大值1023)。
    Args:
        bayer_image_16bit (np.array): 16-bit的拜耳图像数据。
        bayer_pattern_code (int): OpenCV的拜耳模式常量，如 cv2.COLOR_BayerRG2BGR_VNG。
    Returns:
        dict: 包含 'R', 'Gr', 'Gb', 'B' 键的字典，每个值为对应通道的图像矩阵。
    """
    h, w = bayer_image_16bit.shape
    
    R = np.zeros_like(bayer_image_16bit, dtype=np.float32)
    Gr = np.zeros_like(bayer_image_16bit, dtype=np.float32)
    Gb = np.zeros_like(bayer_image_16bit, dtype=np.float32)
    B = np.zeros_like(bayer_image_16bit, dtype=np.float32)
    # *** 关键修改在这里：根据10bit RAW数据，归一化到0-1范围，分母是 1023.0 ***
    bayer_float = bayer_image_16bit.astype(np.float32) / 1023.0 

    if bayer_pattern_code == cv2.COLOR_BayerRG2BGR_VNG: # RGGB
        R[0::2, 0::2] = bayer_float[0::2, 0::2]
        Gr[0::2, 1::2] = bayer_float[0::2, 1::2]
        Gb[1::2, 0::2] = bayer_float[1::2, 0::2]
        B[1::2, 1::2] = bayer_float[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGR2BGR_VNG: # GRBG 
        Gr[0::2, 0::2] = bayer_float[0::2, 0::2]
        R[0::2, 1::2] = bayer_float[0::2, 1::2]
        B[1::2, 0::2] = bayer_float[1::2, 0::2]
        Gb[1::2, 1::2] = bayer_float[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerBG2BGR_VNG: # BGGR
        B[0::2, 0::2] = bayer_float[0::2, 0::2]
        Gb[0::2, 1::2] = bayer_float[0::2, 1::2]
        Gr[1::2, 0::2] = bayer_float[1::2, 0::2]
        R[1::2, 1::2] = bayer_float[1::2, 1::2]
    elif bayer_pattern_code == cv2.COLOR_BayerGB2BGR_VNG: # GBRG
        Gb[0::2, 0::2] = bayer_float[0::2, 0::2]
        B[0::2, 1::2] = bayer_float[0::2, 1::2]
        R[1::2, 0::2] = bayer_float[1::2, 0::2]
        Gr[1::2, 1::2] = bayer_float[1::2, 1::2]
    else:
        raise ValueError("Unsupported Bayer pattern code.")

    return {'R': R, 'Gr': Gr, 'Gb': Gb, 'B': B}
# --- 新增函数：应用增益回拜耳数据 ---
def apply_gain_to_bayer(bayer_image_16bit, gain_map_R, gain_map_Gr, gain_map_Gb, gain_map_B, bayer_pattern_code):
    """
    将单独通道的增益图应用回16-bit拜耳图像。
    注意：这里的增益图是归一化后的增益，需要乘以原始的像素值（在1023范围内）。
    Args:
        bayer_image_16bit (np.array): 原始16-bit拜耳图像。
        gain_map_R, gain_map_Gr, gain_map_Gb, gain_map_B (np.array): 对应通道的增益图。
        bayer_pattern_code (int): OpenCV的拜耳模式常量。
    Returns:
        np.array: 增益补偿后的16-bit拜尔图像。
    """
    compensated_bayer_float = bayer_image_16bit.astype(np.float32) # 转换为浮点进行计算，仍保持0-65535范围

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
    # 限制到16bit范围
    compensated_bayer_16bit = np.clip(compensated_bayer_float, 0, 65535).astype(np.uint16)
    return compensated_bayer_16bit

# --- 【优化后的白平衡函数】 ---
def simple_white_balance(image_rgb_float, mask_2d=None):
    """
    对RGB图像进行稳健的自动白平衡。
    【优化点】: 不再使用整个图像区域，而是使用图像中心的一块矩形区域(例如中心40%x40%的区域)
    来计算R/G/B的平均值。这提供了一个更稳定和准确的“白色”参考，
    不易受到LSC在边缘处可能存在的残留误差的影响，从而提高Chroma Shading的校正效果。

    Args:
        image_rgb_float (np.array): 浮点型RGB图像 (0-1)。
        mask_2d (np.array, optional): 2D掩码。此函数中，它主要用于判断是否需要进行白平衡，
                                      但计算基准始终在中心区域。
    Returns:
        np.array: 白平衡后的RGB图像 (浮点型, 0-1)。
    """
    if image_rgb_float.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB image for white balance.")
    
    # 复制图像以避免修改原始数据
    balanced_image = image_rgb_float.copy()

    h, w, _ = image_rgb_float.shape
    R = balanced_image[:, :, 0]
    G = balanced_image[:, :, 1]
    B = balanced_image[:, :, 2]

    # --- 核心优化：定义一个中心的矩形区域作为白平衡的参考区 ---
    # 使用图像中心 30%-70% 的区域 (即中心40%x40%的区域)
    y_start, y_end = int(h * 0.3), int(h * 0.7)
    x_start, x_end = int(w * 0.3), int(w * 0.7)
    
    # 从原图像中切出中心区域
    central_patch_R = R[y_start:y_end, x_start:x_end]
    central_patch_G = G[y_start:y_end, x_start:x_end]
    central_patch_B = B[y_start:y_end, x_start:x_end]
    
    # 即使提供了mask，我们也只在中心区域计算，因为这里最可靠
    # 如果有掩码，也对掩码进行切片，确保我们只在有效区域内计算
    if mask_2d is not None:
        central_mask_patch = mask_2d[y_start:y_end, x_start:x_end]
        # 结合掩码，找到中心区域内真正有效的像素
        valid_pixels_mask = central_mask_patch > 0.1 # 使用一个小的阈值
        
        # 应用掩码到中心区域
        g_channel_valid = central_patch_G[valid_pixels_mask]
        
        # 检查有效像素是否存在，以及G通道均值是否过低
        if g_channel_valid.size == 0 or np.mean(g_channel_valid) < 1e-6:
            print("Warning: Green channel mean in the central masked area is too low. Skipping white balance.")
            return image_rgb_float # 返回原始图像

        avg_R = np.mean(central_patch_R[valid_pixels_mask])
        avg_G = np.mean(g_channel_valid)
        avg_B = np.mean(central_patch_B[valid_pixels_mask])

    else:
        # 如果没有掩码，直接计算中心区域的平均值
        avg_G = np.mean(central_patch_G)
        if avg_G < 1e-6:
            print("Warning: Average Green channel in the central area is too low. Skipping white balance.")
            return image_rgb_float # 返回原始图
        avg_R = np.mean(central_patch_R)
        avg_B = np.mean(central_patch_B)

    # 防止除以零
    if avg_G < 1e-6:
        print("Warning: Average Green channel is too low for white balance. Skipping white balance.")
        return image_rgb_float
        
    # 计算增益系数
    gain_R = avg_G / (avg_R + 1e-6)
    gain_B = avg_G / (avg_B + 1e-6)
    
    print(f"Robust White Balance Gains (from central patch): R={gain_R:.2f}, G=1.00, B={gain_B:.2f}")

    # 应用增益到整个图像
    balanced_image[:, :, 0] = np.clip(R * gain_R, 0, 1.0)
    balanced_image[:, :, 2] = np.clip(B * gain_B, 0, 1.0)
    
    return balanced_image

# --- 手动选择和调整圆形区域 (修正 'circle_circle' 为 'circle_info') ---
def get_manual_circle_mask(image_rgb_float, feather_pixels, adjust_step=MANUAL_ADJUST_STEP):
    """
    通过交互式鼠标操作，让用户选择和调整圆形区域，并生成羽化掩码。
    Args:
        image_rgb_float (np.array): 用于显示的RGB浮点图像 (0-1)。
        feather_pixels (int): 掩码羽化的像素宽度。
        adjust_step (int): 键盘调整步长。
    Returns:
        np.array: 羽化后的浮点掩码 (0.0-1.0)。
        tuple: 最终确定的圆心 (cx, cy) 和半径 r。
    """
    h, w, _ = image_rgb_float.shape
    display_image = (image_rgb_float * 255).astype(np.uint8) # 用于显示的8-bit图像

    # 将图片缩小到适合屏幕的大小，并保持长宽比
    max_display_dim = 900 # 例如，最大显示尺寸为900像素
    scale = min(max_display_dim / w, max_display_dim / h)
    display_w, display_h = int(w * scale), int(h * scale)
    display_image_resized = cv2.resize(display_image, (display_w, display_h))
    
    # 将颜色空间从RGB转换为BGR，因为OpenCV默认是BGR
    display_image_bgr = cv2.cvtColor(display_image_resized, cv2.COLOR_RGB2BGR)

    current_circle = {'center': None, 'radius': 0}
    drawing = False
    
    window_name = "Select Fisheye Region (Press 'q' to confirm)"

    def draw_circle_on_image(img, circle_info, color=(0, 255, 0), thickness=2):
        temp_img = img.copy()
        # 修正此行：将 'circle_circle' 改为 'circle_info'
        if circle_info['center'] is not None and circle_info['radius'] > 0: 
            # 将缩放后的坐标转换回原始图像坐标
            center_x_orig = int(circle_info['center'][0] / scale)
            center_y_orig = int(circle_info['center'][1] / scale)
            radius_orig = int(circle_info['radius'] / scale)
            
            # 在显示图像上绘制缩放后的圆形
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
                    # 计算半径
                    dx = x - current_circle['center'][0]
                    dy = y - current_circle['center'][1]
                    current_circle['radius'] = int(np.sqrt(dx*dx + dy*dy))
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

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
            cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('s'):
            current_circle['radius'] = max(0, current_circle['radius'] - adjust_step)
            cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('a'):
            if current_circle['center'] is not None:
                current_circle['center'] = (current_circle['center'][0] - adjust_step, current_circle['center'][1])
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('d'):
            if current_circle['center'] is not None:
                current_circle['center'] = (current_circle['center'][0] + adjust_step, current_circle['center'][1])
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('z'):
            if current_circle['center'] is not None:
                current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] - adjust_step)
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('x'):
            if current_circle['center'] is not None:
                current_circle['center'] = (current_circle['center'][0], current_circle['center'][1] + adjust_step)
                cv2.imshow(window_name, draw_circle_on_image(display_image_bgr, current_circle))
        elif key == ord('r'):
            current_circle = {'center': None, 'radius': 0}
            drawing = False
            cv2.imshow(window_name, display_image_bgr.copy())

    cv2.destroyAllWindows()
    # 将缩放后的坐标转换回原始图像坐标
    final_cx = int(current_circle['center'][0] / scale) if current_circle['center'] else w // 2
    final_cy = int(current_circle['center'][1] / scale) if current_circle['center'] else h // 2
    final_r = int(current_circle['radius'] / scale) if current_circle['radius'] > 0 else min(h, w) // 2 - 10

    if final_r <= 0:
        final_r = 1

    hard_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hard_mask, (final_cx, final_cy), final_r, 255, -1)

    # 计算高斯模糊核大小，使其为奇数
    kernel_size = int(2 * feather_pixels / 3) 
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if kernel_size < 3: kernel_size = 3 # 最小核大小为3

    feathered_mask = cv2.GaussianBlur(hard_mask.astype(np.float32), (kernel_size, kernel_size), 0)
    feathered_mask = feathered_mask / 255.0
    feathered_mask = np.clip(feathered_mask, 0.0, 1.0) # 确保掩码值在0-1之间

    return feathered_mask, (final_cx, final_cy, final_r)

# --- 新增函数：绘制和保存热力图 (独立于可视化结果，用于保存每个通道的独立热力图) ---
def plot_heatmap_and_save_matrix(matrix, title_suffix, channel_name, grid_rows, grid_cols, raw_path_for_naming, output_base_dir):
    """
    绘制增益矩阵的热力图并保存到文件。
    动态调整 vmin 和 vmax 以更好地显示通道变化。
    """
    plt.figure(figsize=(10, 8)) # 稍微大一点的图，便于显示
    
    # 寻找非1.0的最小值和最大值，以便更关注衰减区域的变化
    # 增益通常>=1.0，所以我们关注1.0以上的值
    actual_values = matrix[matrix != 1.0] # 筛选出非1.0的增益值
    
    min_display_val = 1.0 
    max_display_val = MAX_GAIN 

    if actual_values.size > 0:
        min_val_calc = np.min(matrix) # 使用整个矩阵的最小值
        max_val_calc = np.max(matrix) # 使用整个矩阵的最大值

        # 根据计算出的实际范围调整显示范围
        # 确保最小值不小于0.1 (增益下限)
        vmin_plot = max(0.1, min_val_calc * 0.95) 
        vmax_plot = max_val_calc * 1.05
        
        # 如果范围太小，强制给定一个有意义的范围
        if abs(vmax_plot - vmin_plot) < 0.01:
            vmin_plot = 1.0
            vmax_plot = max(1.01, vmax_plot) # 至少有0.01的范围

        min_display_val = vmin_plot
        max_display_val = vmax_plot
    
    # 使用 'viridis' 或 'jet' 等颜色映射，通常能更好地显示数值变化
    im = plt.imshow(matrix, cmap='jet', origin='upper', vmin=min_display_val, vmax=max_display_val)
    
    # 添加文本显示每个单元格的值
    for (j, i), val in np.ndenumerate(matrix):
        # 根据值在颜色条中的位置动态调整文本颜色
        # 归一化值到0-1，然后根据颜色映射来决定文本颜色
        normalized_val = (val - min_display_val) / (max_display_val - min_display_val + 1e-6)
        # 简单判断，如果值在颜色条的亮色区域（通常是值高），用黑色文本；否则用白色文本
        # 这取决于具体的colormap，对于'jet'，高值是红色/黄色，低值是蓝色。通常高值用黑色文本更清晰。
        if normalized_val > 0.6: # 阈值可以根据实际效果调整
            text_color = 'black'
        else:
            text_color = 'white'
        
        plt.text(i, j, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0.5, alpha=0.6)) # 添加白色背景，提高可读性
    
    cbar = plt.colorbar(im, label='Gain Value')
    plt.title(f"{channel_name} Channel Gain Map ({title_suffix})") 
    plt.xlabel('Grid Column') 
    plt.ylabel('Grid Row') 
    plt.xticks(np.arange(grid_cols)) 
    plt.yticks(np.arange(grid_rows)) 
    
    # 保存路径
    raw_filename_base = os.path.splitext(os.path.basename(raw_path_for_naming))[0]
    output_dir_heatmaps = os.path.join(output_base_dir, 'heatmaps')
    os.makedirs(output_dir_heatmaps, exist_ok=True)
    
    filename = os.path.join(output_dir_heatmaps, f"{raw_filename_base}_{channel_name}_heatmap.png")
    plt.savefig(filename, bbox_inches='tight') # bbox_inches='tight' 确保所有内容都被保存
    plt.close() # 关闭图表，防止在循环中打开过多窗口
    print(f"已保存 {channel_name} 通道热力图至 {filename}") # 打印内容仍为中文

def save_gain_matrix_to_txt(matrix, channel_name, raw_path_for_naming, output_base_dir, is_golden=False):
    """将增益矩阵保存为文本文件。"""
    raw_filename_base = os.path.splitext(os.path.basename(raw_path_for_naming))[0]
    output_dir_matrices = os.path.join(output_base_dir, 'gain_matrices')
    os.makedirs(output_dir_matrices, exist_ok=True)
    
    if is_golden:
        filename = os.path.join(output_dir_matrices, f"{raw_filename_base}_{channel_name}_golden_table_for_tuning.txt")
        header = f"用于Tuning的Golden增益表 (1024 / script_gain) - {channel_name} 通道:"
        fmt_str = '%d' # 保存为整数
    else:
        filename = os.path.join(output_dir_matrices, f"{raw_filename_base}_{channel_name}_script_gain.txt")
        header = f"脚本计算出的原始增益 - {channel_name} 通道:"
        fmt_str = '%.4f' # 保存为浮点数

    # 将矩阵展平并格式化为单行字符串，使用空格分隔
    flat_matrix = matrix.flatten()
    formatted_line = " ".join([fmt_str % num for num in flat_matrix])

    with open(filename, 'w') as f:
        f.write(f'# {header}\n')
        f.write(formatted_line + '\n')

    print(f"已保存 {header.split('-')[0].strip()} 至: {filename}")


# 【新增】增益平滑函数
def smooth_gain_table(gain_table, kernel_size=3):
    """
    对增益矩阵进行平滑处理以消除异常值。
    Args:
        gain_table (np.array): 原始增益矩阵。
        kernel_size (int): 高斯模糊的核大小，必须为奇数。
    Returns:
        np.array: 平滑后的增益矩阵。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1 # 确保核大小为奇数
    # 使用高斯模糊进行平滑
    smoothed_table = cv2.GaussianBlur(gain_table, (kernel_size, kernel_size), 0)
    return smoothed_table

# 【新增】增益对称化函数
def symmetrize_gain_table(gain_table):
    """
    通过取对称点平均值的方式，强制使增益矩阵中心对称。
    Args:
        gain_table (np.array): 原始增益矩阵。
    Returns:
        np.array: 对称化处理后的增益矩阵。
    """
    rows, cols = gain_table.shape
    symmetrized_table = gain_table.copy()
    
    # 仅需遍历一半的表格即可
    for r in range((rows + 1) // 2):
        for c in range(cols):
            # 找到对称点
            sym_r = rows - 1 - r
            sym_c = cols - 1 - c
            
            # 计算对称点的平均值
            avg_val = (symmetrized_table[r, c] + symmetrized_table[sym_r, sym_c]) / 2.0
            
            # 将平均值赋给对称点
            symmetrized_table[r, c] = avg_val
            symmetrized_table[sym_r, sym_c] = avg_val
            
    return symmetrized_table

# --- 核心校准函数 (主要修改：拜耳数据转换为8-bit用于显示时进行10-bit归一化) ---
def perform_brightness_compensation_with_circle_mask(raw_img_path, width, height, bayer_pattern, 
                                                    grid_rows, grid_cols, 
                                                    feather_pixels=100, max_gain=4.0,
                                                    valid_grid_threshold_ratio=0.05,
                                                    falloff_factor=0.85,
                                                    use_manual_selection=True):
    print(f"--- 开始亮度补偿处理：{os.path.basename(raw_img_path)} ---")
    print(f"图像尺寸: {width}x{height}, 网格大小: {grid_rows}x{grid_cols}")
    
    # 读取 RAW 数据
    original_bayer_16bit = read_raw_bayer_image_manual(raw_img_path, width, height)
    if original_bayer_16bit is None:
        print("错误：无法读取RAW图像。")
        return None, None, None, None

    h, w = original_bayer_16bit.shape
    original_bayer_8bit_for_display = (original_bayer_16bit * (255.0 / 1023.0)).astype(np.uint8)
    original_rgb_float_no_wb = cv2.cvtColor(original_bayer_8bit_for_display, bayer_pattern).astype(np.float32) / 255.0
    temp_display_img_for_selection = simple_white_balance(original_rgb_float_no_wb)

    if use_manual_selection:
        feathered_mask_2d, detected_circle_info = get_manual_circle_mask(temp_display_img_for_selection, feather_pixels, MANUAL_ADJUST_STEP)
        print(f"手动选择已确认: 圆心=({detected_circle_info[0]},{detected_circle_info[1]}), 半径={detected_circle_info[2]}")
    else:
        print("自动霍夫圆检测未在此版本中实现新的拜耳流程。")
        return None, None, None, None

    feathered_mask_3ch = np.stack([feathered_mask_2d] * 3, axis=-1)
    bayer_channels_float = extract_bayer_channels(original_bayer_16bit, bayer_pattern)

   # 根据新的网格行数和列数计算每个网格单元的像素高度和宽度
    # 确保每个网格单元至少有一个像素
    H_grid_cell_size = h // grid_rows
    W_grid_cell_size = w // grid_cols
    H_grid_cell_size = max(H_grid_cell_size, 1)
    W_grid_cell_size = max(W_grid_cell_size, 1)

    grid_brightness_maps = {ch: np.zeros((grid_rows, grid_cols), dtype=np.float32) for ch in ['R', 'Gr', 'Gb', 'B']}
    epsilon = 1e-6
    
    for ch_name, channel_data_sparse in bayer_channels_float.items():
        for i in range(grid_rows): # 遍历行
            for j in range(grid_cols): # 遍历列
                y_start = i * H_grid_cell_size
                y_end = min((i + 1) * H_grid_cell_size, h)
                x_start = j * W_grid_cell_size
                x_end = min((j + 1) * W_grid_cell_size, w)

                grid_area_channel = channel_data_sparse[y_start:y_end, x_start:x_end]
                mask_area = feathered_mask_2d[y_start:y_end, x_start:x_end]
                valid_pixels = (grid_area_channel > epsilon) & (mask_area > epsilon)

                if np.any(valid_pixels):
                    grid_brightness_maps[ch_name][i, j] = np.sum(grid_area_channel[valid_pixels] * mask_area[valid_pixels]) / (np.sum(mask_area[valid_pixels]) + epsilon)
                else:
                    grid_brightness_maps[ch_name][i, j] = 0.0

    # --- 【最终优化算法 V14】: G通道优先 + 衰减因子 + 平滑 + 对称 ---
    # 最终确认：该算法旨在生成能同时校正Luma和Chroma，且不引入色偏的LSC表。
    print("\n--- 开始使用“G通道优先 + 衰减因子 + 平滑 + 对称”算法计算增益 (V14) ---")
    center_row_idx = grid_rows // 2
    center_col_idx = grid_cols // 2

    # 1. 计算理论增益
    G_avg_map = (grid_brightness_maps['Gr'] + grid_brightness_maps['Gb']) / 2.0
    R_map = grid_brightness_maps['R']
    B_map = grid_brightness_maps['B']
    
    center_G_avg = G_avg_map[center_row_idx, center_col_idx]
    center_R = R_map[center_row_idx, center_col_idx]
    center_B = B_map[center_row_idx, center_col_idx]

    if center_G_avg < epsilon:
        print("错误：中心区域G通道亮度过低，无法进行LSC计算。")
        return None, None, None, None

    # Luma gain
    gain_G_raw = np.where(G_avg_map > epsilon, center_G_avg / G_avg_map, 1.0)
    
    # Chroma gain
    target_ratio_R_G = center_R / (center_G_avg + epsilon)
    target_ratio_B_G = center_B / (center_G_avg + epsilon)
    print(f"中心点目标颜色比例: R/G={target_ratio_R_G:.4f}, B/G={target_ratio_B_G:.4f}")

    current_ratio_R_G = np.where(G_avg_map > epsilon, R_map / G_avg_map, 0)
    gain_R_correction = np.where(current_ratio_R_G > epsilon, target_ratio_R_G / current_ratio_R_G, 1.0)
    gain_R_raw = gain_G_raw * gain_R_correction

    current_ratio_B_G = np.where(G_avg_map > epsilon, B_map / G_avg_map, 0)
    gain_B_correction = np.where(current_ratio_B_G > epsilon, target_ratio_B_G / current_ratio_B_G, 1.0)
    gain_B_raw = gain_G_raw * gain_B_correction

    # 2. 应用衰减因子
    print(f"应用衰减因子: {falloff_factor}")
    gain_R_falloff = np.power(gain_R_raw, falloff_factor)
    gain_G_falloff = np.power(gain_G_raw, falloff_factor) # G通道也需要应用
    gain_B_falloff = np.power(gain_B_raw, falloff_factor)
    
    # 3. 平滑处理
    print("对理论增益进行平滑处理...")
    gain_R_smoothed = smooth_gain_table(gain_R_falloff)
    gain_G_smoothed = smooth_gain_table(gain_G_falloff)
    gain_B_smoothed = smooth_gain_table(gain_B_falloff)
    
    # 4. 对称化处理
    print("对平滑后的增益进行对称化处理...")
    gain_R_sym = symmetrize_gain_table(gain_R_smoothed)
    gain_G_sym = symmetrize_gain_table(gain_G_smoothed)
    gain_B_sym = symmetrize_gain_table(gain_B_smoothed)

    # 5. 有效性质询
    validity_threshold = center_G_avg * valid_grid_threshold_ratio
    print(f"中心G通道平均亮度: {center_G_avg:.4f}, 有效性质询门槛: {validity_threshold:.4f}")
    master_valid_mask = G_avg_map > validity_threshold

    # 6. 应用有效性掩码
    gain_R_sym[~master_valid_mask] = 1.0
    gain_G_sym[~master_valid_mask] = 1.0
    gain_B_sym[~master_valid_mask] = 1.0
    
    # 7. 最终裁剪
    script_gain_matrices = {
        'R': np.clip(gain_R_sym, 1.0, max_gain),
        'Gr': np.clip(gain_G_sym, 1.0, max_gain),
        'Gb': np.clip(gain_G_sym, 1.0, max_gain),
        'B': np.clip(gain_B_sym, 1.0, max_gain)
    }
    print("已使用“G通道优先 + 衰减因子 + 平滑 + 对称”机制计算脚本增益矩阵。")


    # --- 保存和可视化增益矩阵 ---
    for ch_name, gain in script_gain_matrices.items():
        # 保存脚本计算出的原始增益
        save_gain_matrix_to_txt(gain, ch_name, raw_img_path, OUTPUT_DIR, is_golden=False)
        
        # 调用热力图生成函数
        plot_heatmap_and_save_matrix(gain, "Script Gain (V14)", ch_name, grid_rows, grid_cols, raw_img_path, OUTPUT_DIR)

        # 计算并保存用于Tuning的Golden增益表
        golden_table = 1024 / (gain + epsilon)
        save_gain_matrix_to_txt(golden_table, ch_name, raw_img_path, OUTPUT_DIR, is_golden=True)

    # 为了可视化，我们仍然使用脚本增益进行补偿
    gain_map_R_full = cv2.resize(script_gain_matrices['R'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gr_full = cv2.resize(script_gain_matrices['Gr'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_Gb_full = cv2.resize(script_gain_matrices['Gb'], (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map_B_full = cv2.resize(script_gain_matrices['B'], (w, h), interpolation=cv2.INTER_LINEAR)

    compensated_bayer_16bit = apply_gain_to_bayer(
        original_bayer_16bit, 
        gain_map_R_full, gain_map_Gr_full, gain_map_Gb_full, gain_map_B_full, 
        bayer_pattern
    )

    compensated_bayer_8bit = (compensated_bayer_16bit * (255.0 / 1023.0)).astype(np.uint8)
    compensated_rgb_float = cv2.cvtColor(compensated_bayer_8bit, bayer_pattern).astype(np.float32) / 255.0
    compensated_rgb_float = compensated_rgb_float * feathered_mask_3ch
    compensated_rgb_float = np.clip(compensated_rgb_float, 0.0, 1.0)

    original_rgb_float_wb = simple_white_balance(original_rgb_float_no_wb, mask_2d=feathered_mask_2d)
    compensated_rgb_float_wb = simple_white_balance(compensated_rgb_float, mask_2d=feathered_mask_2d)

    print("--- 亮度补偿处理完成 ---")
    return original_rgb_float_wb, compensated_rgb_float_wb, original_rgb_float_no_wb, script_gain_matrices


def visualize_results_circle_mask(original_img_wb, compensated_img_wb, original_img_no_wb, gain_data_full_size, output_dir='.'):
    """
    可视化原始图像（白平衡后）、补偿图像（白平衡后）、直方图以及各通道的增益热力图。
    新增：原始未LSC但白平衡的图 vs 做了LSC校正并白平衡的图 对比。
    Args:
        original_img_wb (np.array): 原始去马赛克且白平衡后的RGB图像 (float, 0-1)。
        compensated_img_wb (np.array): LSC补偿且白平衡后的RGB图像 (float, 0-1)。
        original_img_no_wb (np.array): 原始去马赛克但未白平衡的RGB图像 (float, 0-1)。
        gain_data_full_size (dict): 包含R, Gr, Gb, B通道完整尺寸增益图的字典。
        output_dir (str): 输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = os.path.basename(RAW_IMAGE_PATH).split('.')[0]
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
    # 只绘制有效区域的直方图
    valid_original_pixels_no_wb = original_img_no_wb[original_img_no_wb > 0]
    plt.hist(valid_original_pixels_no_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels') # 修改为英文
    plt.title('Original Image Brightness (No WB)') 
    plt.xlabel('Normalized Brightness') 
    plt.ylabel('Pixel Count')
    plt.legend()

    plt.subplot(1, 3, 2)
    valid_original_pixels_wb = original_img_wb[original_img_wb > 0]
    plt.hist(valid_original_pixels_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels') # 修改为英文
    plt.title('Original Image Brightness (with WB)') 
    plt.xlabel('Normalized Brightness') 
    plt.ylabel('Pixel Count') 
    plt.legend()

    plt.subplot(1, 3, 3)
    # 只绘制有效区域的直方图
    valid_compensated_pixels_wb = compensated_img_wb[compensated_img_wb > 0]
    plt.hist(valid_compensated_pixels_wb.flatten(), bins=bins, color='gray', alpha=0.7, label='All Channels') # 修改为英文
    plt.title('Compensated Image Brightness (LSC + WB)') 
    plt.xlabel('Normalized Brightness')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.suptitle('Brightness Distribution Histograms') 
    plt.savefig(os.path.join(output_images_dir, f'{img_name}_brightness_histograms_all.png'), bbox_inches='tight')
    plt.show()

    # 3. 单通道增益热力图 (使用完整尺寸的增益图进行可视化) - 此部分已由 plot_heatmap_and_save_matrix 处理

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
    original_rgb_for_display_wb, compensated_rgb_final_wb, original_rgb_for_display_no_wb, script_gain_matrices = \
        perform_brightness_compensation_with_circle_mask(
            RAW_IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, BAYER_PATTERN,
            grid_rows=GRID_ROWS,
            grid_cols=GRID_COLS,
            feather_pixels=MASK_FEATHER_PIXELS,
            max_gain=MAX_GAIN,
            valid_grid_threshold_ratio=VALID_GRID_THRESHOLD_RATIO,
            falloff_factor=FALLOFF_FACTOR,
            use_manual_selection=USE_MANUAL_CIRCLE_SELECTION
        )

    if original_rgb_for_display_wb is not None:
        base_filename = os.path.basename(RAW_IMAGE_PATH).split('.')[0]
        
        # 保存白平衡且LSC补偿后的PNG图像
        compensated_img_8bit_wb = (compensated_rgb_final_wb * 255).astype(np.uint8)
        output_image_path_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_compensated_feathered_mask_4ch_WB.png')
        cv2.imwrite(output_image_path_wb, cv2.cvtColor(compensated_img_8bit_wb, cv2.COLOR_RGB2BGR))
        print(f"补偿后且白平衡的图像已保存至: {output_image_path_wb}") 

        # 保存原始去马赛克且白平衡后的PNG图像 (用于对比)
        original_img_8bit_wb = (original_rgb_for_display_wb * 255).astype(np.uint8)
        original_output_path_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_original_dem_WB.png')
        cv2.imwrite(original_output_path_wb, cv2.cvtColor(original_img_8bit_wb, cv2.COLOR_RGB2BGR))
        print(f"原始去马赛克且白平衡图像已保存至: {original_output_path_wb}") 

        # 保存原始去马赛克但未白平衡的PNG图像 (用于对比最初的绿色图像)
        original_img_8bit_no_wb = (original_rgb_for_display_no_wb * 255).astype(np.uint8)
        original_output_path_no_wb = os.path.join(OUTPUT_DIR, f'{base_filename}_original_dem_NoWB.png')
        cv2.imwrite(original_output_path_no_wb, cv2.cvtColor(original_img_8bit_no_wb, cv2.COLOR_RGB2BGR))
        print(f"原始去马赛克但未白平衡图像已保存至: {original_output_path_no_wb}") 

        npz_output_path = os.path.join(OUTPUT_DIR, f'{base_filename}_gain_maps_full_size.npz')
        np.savez(npz_output_path,
                 R=script_gain_matrices['R'], 
                 Gr=script_gain_matrices['Gr'], 
                 Gb=script_gain_matrices['Gb'], 
                 B=script_gain_matrices['B'])
        print(f"增益图 (完整尺寸) 已保存至: {npz_output_path}") 
        visualize_results_circle_mask(original_rgb_for_display_wb, compensated_rgb_final_wb, original_rgb_for_display_no_wb, script_gain_matrices, OUTPUT_DIR)
    else:
        print("补偿过程失败。请检查之前的错误信息。")