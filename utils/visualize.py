import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm


def visualize_3d_skeleton(data, attention_scores, time_interval: int = 10):
    """visualize 3d skeleton with attention

    Args:
        data (_type_): skeleton (M T V C)
        attention_scores (_type_): joint attention score (M, T, V)
        time_interval (int, optional): time interval for display. Defaults to 10.
    """
    M, T, _, _ = data.shape
    # 遍历每个人的每一帧,进行可视化
    for j in range(0, T, time_interval):
        # 创建一个3D图形对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 获取当前帧的关节点坐标数据
        joints = data[:, j, :, :]
        scores = attention_scores[:, j, :]

        # 根据注意力分数调整点的颜色和大小
        cmap = cm.get_cmap('Reds')
        sizes = 200 * scores ** 2

        # 绘制关节点之间的连线
        bones = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                (10, 9), (11, 10), (12, 0), (13, 12),
                (14, 13), (15, 14), (16, 0), (17, 16),
                (18, 17), (19, 18), (21, 22), (20, 20),
                (22, 7), (23, 24), (24, 11))

        # 绘制关节点
        for k in range(M):
            color = cmap(scores[k])
            ax.scatter(joints[k, :, 0], joints[k, :, 1], joints[k, :, 2],
                    c=color, s=sizes[k], marker='o')
            for b in bones:
                ax.plot(joints[k, b, 0], joints[k, b, 1], joints[k, b, 2], '-o')

        # 设置坐标轴范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # 显示图形
    plt.show()
