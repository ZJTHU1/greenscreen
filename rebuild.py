#根据得到的节点列表复现图片（有问题）
import cv2
import json
import numpy as np
def draw_contour_from_json(json_path):
    # 读取JSON文件中的坐标数据
    with open(json_path, 'r') as f:
        contour_points = json.load(f)

    # 获取坐标数据中的最大最小坐标值，用于确定图像的尺寸
    x_coords = [point[0] for point in contour_points]
    y_coords = [point[1] for point in contour_points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 创建一个空白的黑色图像，尺寸根据坐标范围确定
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 将坐标数据转换为适合绘制的格式（整数类型的numpy数组）
    contour_points_np = np.array(contour_points, dtype=np.int32)
    # 对坐标进行平移，使其以图像左上角为原点
    contour_points_np[:, 0] -= min_x
    contour_points_np[:, 1] -= min_y

    # 在图像上绘制轮廓
    cv2.drawContours(image, [contour_points_np], -1, (255, 255, 255), 2)

    return image


# 示例用法，替换为你实际保存的JSON文件路径
json_path = "1_contour.json"
result_image = draw_contour_from_json(json_path)
if result_image is not None:
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 可以根据需要保存复现的边界图像，以下是保存为PNG格式的示例
    output_path = json_path.split('.')[0] + "_reconstructed.png"
    cv2.imwrite(output_path, result_image)