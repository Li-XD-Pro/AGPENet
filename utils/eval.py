import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R


def affordance_eval(affordance_list, result):
    """评估功能性检测能力"""
    num_correct = 0
    num_all = 0
    num_points = {aff: 0 for aff in affordance_list}
    num_label_points = {aff: 0 for aff in affordance_list}
    num_correct_fg_points = {aff: 0 for aff in affordance_list}
    num_correct_bg_points = {aff: 0 for aff in affordance_list}
    num_union_points = {aff: 0 for aff in affordance_list}
    num_appearances = {aff: 0 for aff in affordance_list}

    for shape in result:
        for affordance in shape['affordance']:
            label = np.transpose(shape['full_shape']['label'][affordance])
            prediction = shape['result'][affordance][0]

            num_correct += np.sum(label == prediction)
            num_all += 2048
            num_points[affordance] += 2048
            num_label_points[affordance] += np.sum(label == 1.)
            num_correct_fg_points[affordance] += np.sum((label == 1.) & (prediction == 1.))
            num_correct_bg_points[affordance] += np.sum((label == 0.) & (prediction == 0.))
            num_union_points[affordance] += np.sum((label == 1.) | (prediction == 1.))
            num_appearances[affordance] += 1

    # 计算 mIoU，避免除零
    numerator = np.array(list(num_correct_fg_points.values()))
    denominator = np.array(list(num_union_points.values()))
    weights = np.array(list(num_appearances.values()))

    valid_indices = denominator != 0
    if np.any(valid_indices):
        mIoU = np.average(numerator[valid_indices] / denominator[valid_indices],
                          weights=weights[valid_indices] if weights.sum() != 0 else None)
    else:
        mIoU = 0

    Acc = num_correct / num_all
    mAcc = np.mean(
        (np.array(list(num_correct_fg_points.values())) + np.array(list(num_correct_bg_points.values())))
        / (np.array(list(num_points.values())) + 1e-6)  # 避免除零
    )

    return mIoU, Acc, mAcc


def pose_eval(result):
    """评估姿态检测能力"""
    all_min_dist = []
    all_rate = []

    for obj in result:
        for affordance in obj['affordance']:
            gt_poses = np.array([
                np.concatenate((R.from_matrix(p[:3, :3]).as_quat(), p[:3, 3]), axis=0)
                for p in obj['pose'][affordance]
            ])

            # 检查 gt_poses 和检测结果是否为空
            if len(gt_poses) == 0 or len(obj['result'][affordance][1]) == 0:
                continue

            distances = cdist(gt_poses, obj['result'][affordance][1])
            rate = np.sum(np.any(distances <= 0.2, axis=1)) / len(gt_poses)
            all_rate.append(rate)

            min_distance = np.min(distances)
            if min_distance <= 1.0:  # 过滤距离过大的情况
                all_min_dist.append(min_distance)

    return (np.mean(all_min_dist) if all_min_dist else 0,
            np.mean(all_rate) if all_rate else 0)
