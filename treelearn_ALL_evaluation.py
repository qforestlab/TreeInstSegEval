import os
import glob
import laspy

import open3d as o3d
import numpy as np

from fire import Fire
from base_evaluation import BaseEvaluation

class TreeLearnALLEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False, new_version=True):
        # new_version indicates single loop version, initial experiments were with older version with seperate semantic and offset prediction runs
        self.new_version = new_version
        if self.new_version:
            self.method = "treelearn_all"
        else:
            self.method = "treelearn_all_old"
        super(TreeLearnALLEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return

    def read_output(self):
        '''
            Read output TreeLearn
            Output format TreeLearn:
                old version: seperate .npy files with trees
                new version: single .npy file with format x,y,z,instance, instance=0 is ground

            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''
        # use cached predictions if enabled and present
        self.prediction_cache_dir = os.path.join(self.calculation_cache_dir, "predictions", self.dataset, self.method)
        if self.use_cached_calculations:
            if not os.path.exists(self.prediction_cache_dir):
                print(f"INFO: use_cached_calculations is true but predictions are not cached, rereading predictions")
            else:
                return self.read_cached_predictions()

        if self.new_version:
            file = os.path.join(self.output_dir, "results", "full_forest", self.dataset.lower()+"_test.npy")
            arr = np.load(file)
            pc = o3d.t.geometry.PointCloud()
            pc.point.positions = o3d.core.Tensor(arr[:,:3])
            pc.point.instance = o3d.core.Tensor(arr[:, 3][..., np.newaxis])

            predictions = self.seperate_labeled_instances(pc, instance_label="instance", skip_instance=0)
            print(len(predictions))
        else:
            if os.path.exists(os.path.join(self.output_dir, "all")):    
                all_files = glob.glob(os.path.join(self.output_dir, "all", "*.npy"))
            else:
                subfolder1 = os.path.join(self.output_dir, "completely_inside")
                subfolder2 = os.path.join(self.output_dir, "trunk_base_inside")
                subfolder3 = os.path.join(self.output_dir, "trunk_base_outside")

                all_files = glob.glob(os.path.join(subfolder1, "*.npy")) + glob.glob(os.path.join(subfolder2, "*.npy")) + glob.glob(os.path.join(subfolder3, "*.npy"))

            predictions = []
            for file in all_files:
                points = np.load(file)

                pc = o3d.t.geometry.PointCloud()
                pc.point.positions = o3d.core.Tensor(points)

                predictions.append(pc)

        if self.cache_calculations:
            self.cache_predictions(predictions)

        return predictions

if __name__ == "__main__":
    Fire(TreeLearnALLEvaluation)