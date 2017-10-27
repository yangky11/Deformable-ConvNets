from .imdb import IMDB
import os.path
from glob import glob
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import cv2
from utils.show_boxes import show_boxes


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


    def evaluate_detections(self, detections):
        '''writing the results to disk'''
        import pickle
        pickle.dump(detections[1:], open('detections.pickle', 'wb'))
        print(' => results written to detections.pickle')
        for index, filename in enumerate(self.image_filenames):
          im = cv2.imread(filename)
          im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
          show_boxes(im, [det[index] for det in detections[1:]], self.classes[1:], filename.replace('.jpg', '.png'), 1)

