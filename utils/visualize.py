import os
import math
import copy as cp
import pickle

import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import moviepy.editor as mpy

from datasets.pipelines import PreNormalize3D

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

def visualize_2d_skeleton(data, time_interval: int = 10,
                           n_cols=5, axis=[0, 2],
                          figsize=(24, 12), canvas_scale=1,
                          marker='-o', marker_size=4,
                          line_color='gray', point_colors=['red']):
    M, T, V, _ = data.shape
    if len(point_colors) == 1:
        point_colors = point_colors * V
    # 遍历每个人的每一帧，进行可视化
    n_rows = math.ceil(T // time_interval / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    x_min = data[..., axis[0]].min() * canvas_scale
    x_max = data[..., axis[0]].max() * canvas_scale
    y_min = data[..., axis[1]].min() * canvas_scale
    y_max = data[..., axis[1]].max() * canvas_scale
    for i, j in enumerate(range(0, T, time_interval)):
        # 创建一个图形对象
        # fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
        ax_row, ax_col = divmod(i, 5)
        ax = axes[ax_row, ax_col]

        # 获取当前帧的关节点坐标数据
        joints = data[:, j, :, :]

        # 绘制关节点之间的连线
        bones = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20),
                 (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                 (10, 9), (11, 10), (12, 0), (13, 12),
                 (14, 13), (15, 14), (16, 0), (17, 16),
                 (18, 17), (19, 18), (21, 22), (20, 20),
                 (22, 7), (23, 24), (24, 11))

        # 绘制关节点
        for k in range(M):
            # ax.scatter(joints[k, :, 0], joints[k, :, 2],
            #            marker='o', color=color)
            for b in bones:
                ax.plot(joints[k, b, axis[0]], joints[k, b, axis[1]],
                        '-', color=line_color)
            for v in range(V):
                ax.plot(joints[k, v, axis[0]], joints[k, v, axis[1]],
                        'o', markersize=marker_size, color=point_colors[v])
                # ax.plot(joints[k, b, axis[0]], joints[k, b, axis[1]], 'o', markersize=4, color='red')
                # ax.plot(joints[k, b, axis[0]], joints[k, b, axis[1]], '-o')
            # ax.scatter(joints[k, :, 0], joints[k, :, 2],
            #            marker='o', c=colors)

        # 设置坐标轴范围
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.axis('off')

def Vis3DPose(item, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), with_grid=False):
    kp = item['keypoint'].copy()
    colors = ('#3498db', '#000000', '#e74c3c')  # l, m, r

    assert layout == 'nturgb+d'
    if layout == 'nturgb+d':
        num_joint = 25
        kinematic_tree = [
            [1, 2, 21, 3, 4],
            [21, 9, 10, 11, 12, 25], [12, 24],
            [21, 5, 6, 7, 8, 23], [8, 22],
            [1, 17, 18, 19, 20],
            [1, 13, 14, 15, 16]
        ]
        kinematic_tree = [[x - 1 for x in lst] for lst in kinematic_tree]
        colors = ['black', 'blue', 'blue', 'red', 'red', 'darkblue', 'darkred']

    assert len(kp.shape) == 4 and kp.shape[3] == 3 and kp.shape[2] == num_joint
    x, y, z = kp[..., 0], kp[..., 1], kp[..., 2]
    min_x, max_x = min(x[x != 0]), max(x[x != 0])
    min_y, max_y = min(y[y != 0]), max(y[y != 0])
    min_z, max_z = min(z[z != 0]), max(z[z != 0])

    max_axis = max(max_x - min_x, max_y - min_y, max_z - min_z)
    mid_x, mid_y, mid_z = (min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2

    min_x, max_x = mid_x - max_axis / 2, mid_x + max_axis / 2
    min_y, max_y = mid_y - max_axis / 2, mid_y + max_axis / 2
    min_z, max_z = mid_z - max_axis / 2, mid_z + max_axis / 2

    fig = plt.figure(figsize=fig_size)
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([min_x, max_x])
    ax.set_ylim3d([min_y, max_y])
    ax.set_zlim3d([min_z, max_z])
    ax.view_init(*angle)
    fig.suptitle(item.get('frame_dir', 'demo'), fontsize=20)
    save_name = item.get('frame_dir', 'tmp').split('/')[-1] + '.mp4'
    save_path = os.path.join('./demo', save_name)

    def update(t):
        # ax.lines = []
        ax.cla()
        ax.set_xlim3d([min_x, max_x])
        ax.set_ylim3d([min_y, max_y])
        ax.set_zlim3d([min_z, max_z])
        ax.view_init(*angle)
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 2.0
            for j in range(kp.shape[0]):
                ax.plot3D(kp[j, t, chain, 0], kp[j, t, chain, 1], kp[j, t, chain, 2], linewidth=linewidth, color=color)
        if not with_grid:
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=kp.shape[1], interval=0, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()
    video = mpy.VideoFileClip(save_path)
    return video


if __name__ == '__main__':
    annotations = './data/nturgbd/ntu120_3danno.pkl'
    with open(annotations, "rb") as f:
        data = pickle.load(f)
    index = 0
    anno = data['annotations'][index]
    anno = PreNormalize3D()(anno)  # * Need Pre-Normalization before Visualization
    vid = Vis3DPose(anno, layout='nturgb+d', fps=12, angle=(30, 45), fig_size=(8, 8), with_grid=False)