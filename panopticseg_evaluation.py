import os

import open3d as o3d

from fire import Fire
from base_evaluation import BaseEvaluation

class PanopticSegEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        self.method = "panopticseg"
        super(PanopticSegEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return

    def read_output(self):
        '''
            Read output Panoptic Seg method
            Output format Panoptic Seg: Single ply with instances in different color

            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''
        input_file = os.path.join(self.output_dir, "Instance_results_withColor_0.ply")
        input_pc = o3d.t.io.read_point_cloud(input_file)

        predictions = self.seperate_colored_instances(input_pc, remove_zero=False)
        return predictions

if __name__ == "__main__":
    Fire(PanopticSegEvaluation)