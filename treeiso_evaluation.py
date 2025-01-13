import os

import open3d as o3d

from fire import Fire
from base_evaluation import BaseEvaluation

class TreeIsoEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False, output_file="final/result.ply"):
        self.method="treeiso"
        self.output_file = output_file
        super(TreeIsoEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return


    def read_output(self):
        '''
            Read output TreeIso Seg method

            Output format TreeIso Seg:
                Single ply with instances marked by final segs

            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''

        input_file = os.path.join(self.output_dir, self.output_file)
        
        input_pc = o3d.t.io.read_point_cloud(input_file)

        predictions = self.seperate_labeled_instances(input_pc, instance_label="scalar_final_segs")

        return predictions

if __name__ == "__main__":
    Fire(TreeIsoEvaluation)