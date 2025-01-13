import os
import glob

import open3d as o3d
import numpy as np

from fire import Fire
from evaluation.evaluation_classes.base_evaluation import BaseEvaluation

class Mask3DEvaluation(BaseEvaluation):

    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        self.method = "mask3d"
        super(Mask3DEvaluation, self).__init__(data_base_dir, dataset, use_cached_calculations, cache_calculations, debug)
        return


    def read_output(self, tile_name=None):
        '''
            Read output Mask3D

            Output format Mask3D:

                # file per tile with all instance masks, read these and index pointcloud to get instances


            Output format:
                [tree1, tree2, ...] where treen is an o3d instance of tree
        '''
        # TODO: try for Mask3D

        # mask3d runs on tiles, so we need some way to merge the instances found across tiles
        # perhaps we could also try if it runs on whole test area at once


        for tile in tiles:
            # read in tile

            # seperate instances based on masks
            instances = self.seperate_masked_instances(masks, tile_pc)

        
        self.merge_tiled_instances(tiled_instances, buffer_size=5)



        raise NotImplementedError

    def test_eval(self):

        predictions = self.read_output(tile_name="Wytham_Tile0")

        gt_instances_eval, gt_instances_no_eval = self.read_gt(tile_name="Wytham_Tile0", thresholded=True)

        self.eval(predictions, gt_instances_eval, gt_instances_no_eval, debug=True)

        return


if __name__ == "__main__":
    Fire(Mask3DEvaluation)