import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from tqdm import tqdm

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [json.loads(line) for line in lines]

def fit_line_ransac(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    line = ransac.estimator_

    return line.coef_[0], line.intercept_

def process_data(data):
    dataset = []
    for frame_data in tqdm(data, desc="Processing Frames"):
        lanes = frame_data['lanes']
        lines = []
        for points in lanes:
            points_array = np.array(points)
            line = fit_line_ransac(points_array)
            lines.append(line)
        dataset.append(lines)
    return dataset

def cal_offset_from_marks(dataset, method=1):
    dd = np.zeros(len(dataset))
    for i, lanemarks in enumerate(dataset):
        sortmarks, idx = search_focus_loc(lanemarks, 1080)
        if idx == -1:
            continue

        left = sortmarks[idx]
        right = sortmarks[idx + 1]
        kl = left[0]
        kr = right[0]

        if method == 1:
            c = kl / (kr - kl)    
            if kl == float('inf'):
                c = 1
            elif kr == float('inf'):
                c = 0

        dd[i] = abs(c)

    return dd

def search_focus_loc(lanemarks, mid):
    lnum = len(lanemarks)
    sortmarks = []
    if lnum <= 1:
        return sortmarks, -1

    loc = np.zeros(lnum)
    for il, lane in enumerate(lanemarks):
        lp = lane
        loc[il] = (1200 - lp[1]) / lp[0]

    index = np.argsort(loc)
    sortmarks = [lanemarks[i] for i in index]

    for j in range(lnum - 1):
        if loc[index[j]] < mid and loc[index[j + 1]] > mid:
            return sortmarks, j

    return sortmarks, -1

def plot_offset(dd, save_path):
    plt.plot(dd, '.', color=[0.57, 0.12, 0.57])
    x = np.linspace(-500, 3000, 300)
    for y in [0, 1, 2, -1]:
        plt.plot(x, np.full_like(x, y), '--k', linewidth=2)
    plt.savefig(save_path)
    plt.close()


def calculate_global_position(dd):
    n = len(dd)
    global_pos = np.zeros(n)
    for k in range(1, n):
        cur = dd[k]
        pre = dd[k - 1]
        if cur - pre <= -0.8:
            global_pos[k] = global_pos[k - 1] + 1
        elif cur - pre >= 0.8:
            global_pos[k] = global_pos[k - 1] - 1
        else:
            global_pos[k] = global_pos[k - 1]

    for k in range(n):
        dd[k] += global_pos[k]

    return dd

if __name__ == "__main__":
    file_path = "lanes.json"
    save_path = "lane_offset_plot.png"

    data = load_data(file_path)
    dataset = process_data(data)
    dd = cal_offset_from_marks(dataset)
    dd = calculate_global_position(dd)
    plot_offset(dd, save_path)