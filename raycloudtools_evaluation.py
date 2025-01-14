import os
import glob
import re

import open3d as o3d

from fire import Fire
from base_evaluation import BaseEvaluation

class RayCloudToolsEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        self.method = "raycloudtools"
        super(RayCloudToolsEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return

    def read_output(self):
        '''
            Read output RayCloudTools
            Output format RayCloudTools: single ply with colored instances, color (0,0,0) is ground

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
        
        # if not cached, read in predictions
        FILENAME = self.dataset.lower() + "_test_raycloud_segmented.ply"
        input_file = os.path.join(self.output_dir, FILENAME)
        input_pc = o3d.t.io.read_point_cloud(input_file)

        predictions = self.seperate_colored_instances(input_pc, remove_zero=True)

        if self.cache_calculations:
            self.cache_predictions(predictions)
        
        return predictions

if __name__ == "__main__":
    Fire(RayCloudToolsEvaluation)