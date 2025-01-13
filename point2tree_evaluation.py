import os
import glob

import open3d as o3d
import numpy as np

from fire import Fire
from base_evaluation import BaseEvaluation

class Point2TreeEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        self.method = "point2tree"
        super(Point2TreeEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return


    def read_output(self):
        '''
            Read output Point2tree

            Output format Point2tree:
                ODIR/instance_segmented_point_clouds/dataset.instance_segmented.ply

                labels are under "instance_nr"

            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''

        input_file = os.path.join(self.output_dir,"instance_segmented_point_clouds", self.dataset.lower()+".instance_segmented.ply")

        
        input_pc = o3d.t.io.read_point_cloud(input_file)

        predictions = self.seperate_labeled_instances(input_pc, instance_label="instance_nr")

        return predictions


if __name__ == "__main__":
    Fire(Point2TreeEvaluation)