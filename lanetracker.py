import os
import numpy as np
import cv2

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0])
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def similarity_batch(detected_lanes, tracked_lanes, distance_weight=0.1, angle_weight=1.0, length_weight=0.05):
    num_detections = detected_lanes.shape[0]
    num_trackers = tracked_lanes.shape[0]
    similarity_matrix = np.zeros((num_detections, num_trackers), dtype=np.float32)
    
    for d in range(num_detections):
        for t in range(num_trackers):
            distance = np.linalg.norm(detected_lanes[d, :2] - tracked_lanes[t, :2])
            
            angle_diff = np.abs(detected_lanes[d, 2] - tracked_lanes[t, 2])
            angle_diff = min(angle_diff, 1 - angle_diff)
            
            length_diff = np.abs(detected_lanes[d, 3] - tracked_lanes[t, 3])
            
            similarity = (distance_weight / (distance + 1e-6)) + \
                         (angle_weight / (angle_diff + 1e-6)) + \
                         (length_weight / (length_diff + 1e-6))
            
            similarity_matrix[d, t] = similarity
    
    return similarity_matrix

class KalmanBoxTracker(object):
  count = 0
  def __init__(self,detection):
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10
    self.kf.P[4:,4:] *= 1000
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:6,4:6] *= 0.01

    x, y, angle, length, _ = detection
    self.kf.x[:4] = np.array([[x], [y], [angle], [length]])
    self.kf.x[4:] = 0

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, detection):
      self.time_since_update = 0
      self.history = []
      self.hits += 1
      self.hit_streak += 1
      self.kf.update(detection[:4])

  def predict(self):
      if ((self.kf.x[6] + self.kf.x[2]) <= 0):
          self.kf.x[6] *= 0.0
      self.kf.predict()
      self.age += 1
      if (self.time_since_update > 0):
          self.hit_streak = 0
      self.time_since_update += 1
      self.history.append(self.kf.x[:4])
      return self.history[-1]

  def get_state(self):
      return self.kf.x[:4]


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = similarity_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    dets = np.array(dets)
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict().flatten()
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state().flatten()
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
        i -= 1
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def load_data_with_variable_points(filename, delimiter=','):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(delimiter)
            fixed_data = list(map(float, parts[:10]))
            control_points = parts[10:]
            if len(control_points) % 2 != 0:
                print("Warning: Number of control points in row data is not pairwise")
                continue
            control_points = [(int(control_points[i]), int(control_points[i + 1])) for i in range(0, len(control_points), 2)]
            data.append(fixed_data + [control_points])
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(1000)}

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold)
    seq_dets = load_data_with_variable_points(seq_dets_fn)
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      min_frame_id = int(min(det[0] for det in seq_dets))
      max_frame_id = int(max(det[0] for det in seq_dets))
      total_frames = 0

      for frame in range(min_frame_id, max_frame_id + 1):
        dets = [det[2:7] for det in seq_dets if det[0] == frame]
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          track_id = int(d[4])
          x = d[0]
          y = d[1]
          angle = d[2]
          length = d[3]
          confidence = 1
          print(f'{frame},{track_id},{x:.2f},{y:.2f},{angle:.4f},{length:.2f},{confidence},-1,-1,-1', file=out_file)

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    for frame in range(min_frame_id, max_frame_id + 1):
        fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg' % frame)
        img = cv2.imread(fn)
        
        dets = [det[2:7] for det in seq_dets if det[0] == frame]
        trackers = mot_tracker.update(dets)
        
        for d in trackers:
            track_id = int(d[4])
            points = d[5] 
            color = colors[track_id] if track_id in colors else (255, 255, 255)

            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i - 1]))
                pt2 = tuple(map(int, points[i]))
                cv2.circle(img, pt1, 2, color, -1)
                cv2.line(img, pt1, pt2, color, 2)
                
        cv2.imwrite(os.path.join('output', '%s_%06d.jpg' % (seq, frame)), img)

    print("Note: to get real runtime results run without the option: --display")
