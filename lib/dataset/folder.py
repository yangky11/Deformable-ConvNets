from .imdb import IMDB
import os.path
from glob import glob


class Folder(IMDB):

    def __init__(self, data_path):
        self.data_path = data_path
        root_path = os.path.split(data_path)[0]
        result_path = os.path.join(root_path, 'result')
        super().__init__('FOLDER', 'test', root_path, data_path, result_path)
        self.image_filenames = list(glob(os.path.join(data_path, '*.jpg')))
        print(' => creating dataset from %s' % data_path)
        print(' => %d images' % len(self.image_filenames))


    def image_path_from_index(self, index):
      return self.image_filenames[index] 


    def gt_roidb(self):
      return []


    def evaluate_detections(self, detections):
        '''writing the results to disk'''
        import pickle
        pickle.dump(detections, open('detections.pickle', 'wb'))
        print(' => results written to detections.pickle')

