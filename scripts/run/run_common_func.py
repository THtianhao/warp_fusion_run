import os
import shutil
from datetime import datetime

from scripts.utils.env import root_dir
from scripts.video_process.video_config import VideoConfig

def get_scheduled_arg(frame_num, schedule):
    if isinstance(schedule, list):
        return schedule[frame_num] if frame_num < len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
        return get_sched_from_json(frame_num, schedule, blend=False)

def get_sched_from_json(frame_num, sched_json, blend=False):
    frame_num = max(frame_num, 0)
    sched_int = {}
    for key in sched_json.keys():
        sched_int[int(key)] = sched_json[key]
    sched_json = sched_int
    keys = sorted(list(sched_json.keys()))
    # print(keys)
    if frame_num < 0:
        frame_num = max(keys)
    try:
        frame_num = min(frame_num, max(keys))  # clamp frame num to 0:max(keys) range
    except:
        pass

    # print('clamped frame num ', frame_num)
    if frame_num in keys:
        return sched_json[frame_num]
        # print('frame in keys')
    if frame_num not in keys:
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i + 1]
            if frame_num > k1 and frame_num < k2:
                if not blend:
                    print('frame between keys, no blend')
                    return sched_json[k1]
                if blend:
                    total_dist = k2 - k1
                    dist_from_k1 = frame_num - k1
                    return sched_json[k1] * (1 - dist_from_k1 / total_dist) + sched_json[k2] * (dist_from_k1 / total_dist)
            # else: print(f'frame {frame_num} not in {k1} {k2}')
    return 0

def printf(*msg, file=f'{root_dir}/log.txt'):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(file, 'a') as f:
        msg = f'{dt_string}> {" ".join([str(o) for o in (msg)])}'
        print(msg, file=f)

import shutil
import os

def copy_and_rename_file(source_file, destination_folder, new_filename):
    try:
        # 确保源文件存在
        if not os.path.exists(source_file):
            raise FileNotFoundError("源文件不存在")

        # 确保目标文件夹存在
        os.makedirs(destination_folder, exist_ok=True)

        # 构建目标文件路径
        destination_file = os.path.join(destination_folder, new_filename)

        # 使用shutil.copy2复制并重命名文件
        shutil.copy2(source_file, destination_file)

        print("文件复制和重命名完成！")
    except Exception as e:
        print("出现错误:", str(e))
