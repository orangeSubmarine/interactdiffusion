import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def print_boundingbox(res):

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.invert_yaxis()  # 图像坐标系（原点在左上角）

    # 绘制边界框
    for subject_box, object_box, action in zip(res["subject_boxes"],res["object_boxes"],res["action_phrases"]):
        for i in range(len(subject_box)):
            subject_box[i]=subject_box[i]*300
        for i in range(len(object_box)):
            object_box[i]=object_box[i]*300
        rect1 = patches.Rectangle(
            (subject_box[0], subject_box[1]), subject_box[2], subject_box[3],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        rect2 = patches.Rectangle(
            (object_box[0], object_box[1]), object_box[2], object_box[3],
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        # 计算中心点
        center1 = (subject_box[0] + subject_box[2]/2, subject_box[1] + subject_box[3]/2)
        center2 = (object_box[0] + object_box[2]/2, object_box[1] + object_box[3]/2)

        # 绘制连线
        ax.plot(
            [center1[0], center2[0]],
            [center1[1], center2[1]],
            color='green', linestyle='--', linewidth=2
        )

        # 添加文字注释
        # ax.text(subject_box[0], subject_box[1]-10, 'subject_box', 
        #         color='red', fontsize=10, weight='bold')
        # ax.text(object_box[0], object_box[1]-10, 'object_box', 
        #         color='blue', fontsize=10, weight='bold')
        # ax.text((center1[0]+center2[0])/2, (center1[1]+center2[1])/2, 
        #         action, color='green', fontsize=10, ha='center')
        ax.text(100, 0, 
    res["prompt"], color='green', fontsize=10, ha='center')

    plt.savefig(os.path.join("/media/store/lxc/papers/Interactdiffusion/interactdiffusion/generation_samples/bbox",res['file_name']))
    plt.close()