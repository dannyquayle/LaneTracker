import json
import math

def calculate_average_angle(xys, strip_size):
    xs_inside_image = [point[0] for point in xys]
    thetas = []

    for i in range(1, len(xs_inside_image)):
        theta = math.atan(i * strip_size / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
        theta = theta if theta > 0 else 1 - abs(theta)
        thetas.append(theta)

    theta_far = sum(thetas) / len(thetas)
    return round(theta_far, 4)


def convert_json_line_to_det_format(json_line, strip_size):
    frame_data = json.loads(json_line)
    det_lines = []

    frame_id = extract_frame_id(frame_data["image_path"])
    for lane in frame_data["lanes"]:
        track_id = -1
        start_x, start_y = map(int, lane["xys"][0]) 
        angle = calculate_average_angle(lane["xys"], strip_size)
        length = len(lane["xys"])
        confidence = lane["cls_score"]
        extra_cols = -1

        points_str = ",".join(",".join(map(str, map(int, point))) for point in lane["xys"])

        det_line = f"{frame_id},{track_id},{start_x},{start_y},{angle},{length},{confidence},{extra_cols},{extra_cols},{extra_cols},{points_str}"
        det_lines.append(det_line)

    return det_lines


def read_json_convert_to_det(json_file, strip_size):
    all_det_lines = []
    with open(json_file, 'r') as file:
        for line in file:
            det_lines = convert_json_line_to_det_format(line, strip_size)
            all_det_lines.extend(det_lines)
    return all_det_lines


def extract_frame_id(image_path):
    return int(image_path.split('/')[-1].split('.')[0])


def save_det_file(det_lines, output_file):
    with open(output_file, 'w') as file:
        for line in det_lines:
            file.write(line + '\n')



if __name__ == '__main__':

    json_file = 'Lanes.json'
    output_file = 'det.txt'
    strip_size = 10
    det_lines = read_json_convert_to_det(json_file, strip_size)

    save_det_file(det_lines, output_file)
