#处理绿幕 得到边界节点列表
import cv2
import numpy as np
import json
def remove_green_screen(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径是否正确")
        return
    # 将图像转换到HSV色彩空间，便于更好地处理颜色信息
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 定义绿幕的HSV范围（这里的范围是大致的，可以根据实际情况调整）
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])

    # 创建绿幕的掩膜，将在这个范围内的像素设为白色（255），其他设为黑色（0）
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 计算掩膜的梯度，通过Sobel算子分别计算x和y方向的梯度
    sobelx = cv2.Sobel(green_mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(green_mask, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_magnitude = np.uint8(gradient_magnitude)

    # 对梯度进行阈值处理，以便更清晰地区分边界
    _, thresholded_gradient = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    # 结合绿幕掩膜和梯度阈值化的结果，完善抠图的掩膜
    final_mask = cv2.bitwise_and(green_mask, thresholded_gradient)

    # 使用最终的掩膜来抠取图像中的内容
    result = cv2.bitwise_and(image, image, mask=final_mask)

    # 查找边界轮廓
    contours, _ = cv2.findContours(final_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_contour_points = []
    for contour in contours:
        # 将每个轮廓的节点坐标转换为列表形式，并存储到新的列表中，顺序保持不变
        contour_points = contour.reshape(-1, 2).tolist()
        all_contour_points.extend(contour_points)
    return all_contour_points


# 示例用法，替换为你实际的绿幕图像路径
image_path = "1.png"
contour_points = remove_green_screen(image_path)
if contour_points:
    # 构建保存JSON文件的文件名，这里示例为在原文件名基础上加"_contour.json"后缀
    json_output_path = image_path.split('.')[0] + "_contour.json"
    with open(json_output_path, 'w') as f:
        json.dump(contour_points, f)
    print(f"边界节点坐标已保存至 {json_output_path}")