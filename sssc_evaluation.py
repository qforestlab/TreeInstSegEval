import os

import open3d as o3d

from fire import Fire
from base_evaluation import BaseEvaluation

class SSSCEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        self.method="sssc"
        super(SSSCEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return


    def read_output(self):
        '''
            Read output SSSC Seg method

            Output format SSSC Seg:
                Single ply with instances under instance label

            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''

        input_file = os.path.join(self.output_dir, self.dataset.lower()+"_segmented_final.ply")
        
        input_pc = o3d.t.io.read_point_cloud(input_file)

        predictions = self.seperate_labeled_instances(input_pc, instance_label="instance")

        return predictions

if __name__ == "__main__":
    Fire(SSSCEvaluation)