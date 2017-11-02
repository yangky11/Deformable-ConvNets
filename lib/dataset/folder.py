from .imdb import IMDB
import os.path
from glob import glob
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import cv2
from utils.show_boxes import show_boxes
import matplotlib.pyplot as plt
import pickle
import random


class folder(IMDB):

  def __init__(self, data_path):
    self.data_path = data_path
    root_path = os.path.split(data_path)[0]
    result_path = os.path.join(root_path, 'result')
    super().__init__('FOLDER', 'test', root_path, data_path, result_path)

    self.image_filenames = list(glob(os.path.join(data_path, '*.jpg')))
    self.coco = COCO('/scratch/jiadeng_fluxoe/shared/cocoapi/data/annotations/image_info_test2017.json')
    cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
    self.classes = ['__background__'] + cats
    self.num_classes = len(self.classes)
    self.num_images = len(self.image_filenames)

    print(' => creating dataset from %s' % data_path)
    print(' => %d images' % self.num_images)


  def image_path_from_index(self, index):
    return self.image_filenames[index] 


  def draw_region_proposals(self, im, rois, thresh, filename):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for i in range(rois['bbox'].shape[0]):
      bbox = rois['bbox'][i]
      score = rois['score'][i]
      if score < thresh:
        continue
      color = (random.random(), random.random(), random.random())
      rect = plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor=color, linewidth=2.5)
      plt.gca().add_patch(rect)
      plt.gca().text(bbox[0], bbox[1],
                     '%.3f' % score,
                     bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    plt.savefig(filename)
    return im


  def gt_roidb(self):
    roidb = []
    for index, filename in enumerate(self.image_filenames):
      width, height = Image.open(filename).size
      roidb.append({'image': filename,
                    'height': height,
                    'width': width,
                    'boxes': np.zeros((0, 4), dtype=np.uint16),
                    'flipped': False})
    return roidb


  def evaluate_detections(self, detections, rois):
    '''writing the results to disk'''
    pickle.dump(detections[1:], open(os.path.join(self._result_path, 'detections.pickle'), 'wb'))
    pickle.dump(rois, open(os.path.join(self._result_path, 'rois.pickle'), 'wb'))
    print(' => results written to %s' % self._result_path)
    for index, filename in enumerate(self.image_filenames):
      im = cv2.imread(filename)
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      show_boxes(im, [det[index] for det in detections[1:]], self.classes[1:], filename.replace('.jpg', '_objects.png'), 1)
      self.draw_region_proposals(im, rois[index], 0.5, filename.replace('.jpg', '_proposals.png'))
