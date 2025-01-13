import os
import glob
import re

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from fire import Fire
from scipy.optimize import linear_sum_assignment
from timeit import default_timer as timer

DATASETS = ["WYTHAM", "LITCHFIELD", "OFENTAL", "ROBSONCREEK"]


class BaseEvaluation():
    
    def __init__(self, data_base_dir= "/Stor1/wout/BenchmarkPaper/data/", dataset="WYTHAM", use_cached_calculations=True, cache_calculations=True, debug=False):
        # check if dataset supported
        if dataset not in DATASETS:
            print(f"Dataset {dataset} not found in available datasets {DATASETS}")
            print(f"If adding a new evaluation dataset, make sure BASE data is available at gt_dir/{dataset} and add the dataset to the list at the top of base_evaluation.py")
            os._exit(1)

        # init variables
        self.data_base_dir = data_base_dir
        self.dataset = dataset

        self.use_cached_calculations = use_cached_calculations
        self.cache_calculations = cache_calculations
        self.debug = debug

        # gt data
        # TODO: check if all required folders in gt_dir/dataset?
        self.gt_dir = os.path.join(self.data_base_dir, "BASE", self.dataset)
        if not os.path.exists(self.gt_dir):
            print(f"Can't find ground truth folder {self.gt_dir}")
            print(f"Required folder structure for: data_base_dir/BASE/dataset/...")
            os._exit(1)

        # output dir: detect based on method name
        for dir in glob.glob(os.path.join(self.data_base_dir, "OUTPUTS", "*/"), recursive=True):
            if os.path.isdir(dir) and os.path.basename(os.path.basename(os.path.normpath(dir.lower()))) == self.method:
                self.output_dir = dir
                break
        
        if self.output_dir is None:
            print(f"Cant find output folder for method {self.method} and dataset {self.dataset}")
            print(f"Required folder structure: data_base_dir/OUTPUTS/method/DATASET/..")
            os._exit(1)

        self.output_dir = os.path.join(self.output_dir, self.dataset)
        if not os.path.exists(self.output_dir):
            print(f"Can't find output folder {self.output_dir}")
            print(f"Required folder structure: data_base_dir/OUTPUTS/method/DATASET/..")
            os._exit(1)

        # output dir for evaluation results
        self.eval_output_dir = os.path.join(self.data_base_dir, "EVALUATION", self.method, self.dataset)
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)

        # calculation cache dir
        self.calculation_cache_dir = os.path.join(self.data_base_dir, "calculation_cache")

        print("-----------------------------")
        print(f"Running evaluation for method {self.method} on dataset {self.dataset}")
        print(f"Ground truth data directory: {self.gt_dir}")
        print(f"Prediction data directory: {self.output_dir}")
        print(f"Evaluation output directory: {self.eval_output_dir}")
        print(f"Data cache directory: {self.calculation_cache_dir} (use_cached_calculations={self.use_cached_calculations}, cache_calculations={cache_calculations})")
        print(f"Debug is set to {self.debug}")
        print("-----------------------------")


        # read gt and predictions
        self.prepare_inputs()


    def read_gt(self, tile_name=None, thresholded=True):
        '''
            Read ground truth trees.

            If a tile_name is provided, only reads this data for single tile
            
            If tiled, output is dict[tilename] = tile where tile is o3d instance
            Otherwise, output is o3d instances of test, val and train plot
        '''

        source_dir = None
        if tile_name is not None:
            source_dir = os.path.join(self.gt_dir, "trees", "test", tile_name)
        else:
            source_dir = os.path.join(self.gt_dir, "test_trees_thresholded")

        if not os.path.exists(source_dir):
            print(f"Couldn't find path {dir}")
            return
        
        # we read in bbox of entire test_area, as some eval trees have small part outside area that can never be part of prediction so should be cut of
        test_area = o3d.t.io.read_point_cloud(os.path.join(self.gt_dir, self.dataset.lower() + "_test_merged.ply"))
        bbox_test_area = test_area.get_axis_aligned_bounding_box()
        
        if not thresholded: # shouldn't really ever be used, maybe for test
                trees = []
                tree_names = []
                for file in glob.glob(os.path.join(source_dir, "*.ply")):
                    tree_name = os.path.basename(file)[:-4]
                    tree = o3d.t.io.read_point_cloud(file)
                    # crop with bbox of area
                    tree = tree.crop(bbox_test_area)
                    trees.append(tree)
                    tree_names.append(tree_name)
                return trees, [], tree_names # second argument is non-eval trees
        else:
            trees_eval = []
            trees_non_eval = []
            tree_names = []
            for file in glob.glob(os.path.join(source_dir, "in_plot_*", "*.ply")):
                tree_name = os.path.basename(file)[:-4]
                tree = o3d.t.io.read_point_cloud(file)
                # crop with bbox of area
                tree = tree.crop(bbox_test_area)
                trees_eval.append(tree)
                tree_names.append(tree_name)
            for file in glob.glob(os.path.join(source_dir, "out_plot_*", "*.ply")):
                tree = o3d.t.io.read_point_cloud(file)
                # crop with bbox of area
                tree = tree.crop(bbox_test_area)
                trees_non_eval.append(tree)
            return trees_eval, trees_non_eval, tree_names

    def read_output(self, debug=False):
        print("ERROR: reading output is method dependent and should be defined in each method")
        raise NotImplementedError("read_output is method-specific")

        if debug:
            print("WARNING: reading output is method-specific and not implemented!!")
            print("For debugging using base method, we read in ground truth test trees")

            trees_dir = os.path.join(self.gt_dir, "trees")

            test_trees_dir = os.path.join(trees_dir, "test")

            out_test = {}
            for file in glob.glob(os.path.join(test_trees_dir, "*.ply")):
                tn = os.path.splitext(os.path.basename(file))
                out_test[tn] = o3d.t.io.read_point_cloud(file)

            return out_test
        else:
            print("ERROR: reading output is method dependent and should be defined in each method")
            raise NotImplementedError("read_output is method-specific")

    def prepare_inputs(self):
        print("Reading predictions")
        self.predictions = self.read_output()

        print("Reading ground truth instances")
        self.gt_eval, self.gt_no_eval, self.tree_names = self.read_gt(thresholded=True)
        return


    def seperate_labeled_instances(self, pc, instance_label="instance", skip_instance=-1):
        '''
            Seperates pointcloud into array of instances based on instance_label (default=instance)
        '''

        points = pc.point.positions.numpy()
        instance = pc.point[instance_label].numpy()

        trees = []

        for instance_number in np.unique(instance):
            if instance_number == skip_instance or np.isnan(instance_number):
                # ground points or NaN, skip
                continue
            
            # get mask to select elements of arrays that are part of tree
            instance_mask = (instance == instance_number).flatten()

            tree = o3d.t.geometry.PointCloud()
            tree.point.positions = points[instance_mask]
            tree.point.instance = instance[instance_mask]

            trees.append(tree)

        return trees

    def seperate_colored_instances(self, pc, remove_zero=False):
        '''
            Seperate pointcloud based on color
        '''


        points = pc.point.positions.numpy()
        colors = pc.point.colors.numpy()

        unique_colors = np.unique(colors, axis=0)

        instances = []
        for color in unique_colors:
            # color 0,0,0 are all points classified as non-instances NOTE: this is for raycloudtools, might be different for other methods, give as arguments
            if remove_zero and (color == np.array([0,0,0])).all():
                continue

            idx_mask = np.all(colors == color, axis=1)

            tree_points = points[idx_mask]
            tree = o3d.t.geometry.PointCloud(tree_points)
            instances.append(tree)

        return instances

    def read_cached_predictions(self):
        regex = re.compile(r'\d+')
        prediction_files = sorted(glob.glob(os.path.join(self.prediction_cache_dir, "*.ply")), key=lambda x:int(regex.findall(x)[-1]))
        print(f"Reading {len(prediction_files)} predictions from {self.prediction_cache_dir}")
        predictions = []
        for file in prediction_files:
            pc = o3d.t.io.read_point_cloud(file)
            predictions.append(pc)
        return predictions

    def cache_predictions(self, predictions):
        print(f"Caching predictions at {self.prediction_cache_dir}")
        if not os.path.exists(self.prediction_cache_dir):
            os.makedirs(self.prediction_cache_dir, exist_ok=True)
        for i, prediction in enumerate(predictions):
            o3d.t.io.write_point_cloud(os.path.join(self.prediction_cache_dir, f"prediction_{i}.ply"), prediction)

    def eval(self, odir=None):
        '''
            Performs general eval of instance segmentation.

            All three arguments should be lists of o3d.t.geometry.Pointcloud instances
        '''

        print("")
        print(f"Performing evaluation using {len(self.predictions)} predictions, {len(self.gt_eval)} (eval) + {len(self.gt_no_eval)} (no_eval) ground truth instances.")
        if odir is not None:
            os.makedirs(odir, exist_ok=True)
        else:
            odir=self.eval_output_dir

        ## 1. Calculate IoU for each prediction and gt (or use cached as this step takes a long time)

        if not self.use_cached_calculations:
            print("")
            print(f"Calculating IoU")

            all_gt_instances = self.gt_eval + self.gt_no_eval

            start = timer()
            IoU_arr = self.calculate_IoU(self.predictions, all_gt_instances, debug=self.debug)
            end = timer()
            print(f"Running time to calc IoU: {end - start:.3f} s")

            if self.cache_calculations:
                IoU_cache_dir = os.path.join(self.calculation_cache_dir, "IoU_cache", self.dataset)
                if not os.path.exists(IoU_cache_dir):
                    os.makedirs(IoU_cache_dir)
                path = os.path.join(IoU_cache_dir, f"{self.method}_IoU_arr.npy")
                print(f"Saving IoU array at {path}")
                np.save(path, IoU_arr)
        else:
            print("")
            IoU_cache_dir = os.path.join(self.calculation_cache_dir, "IoU_cache", self.dataset)
            path = os.path.join(IoU_cache_dir, f"{self.method}_IoU_arr.npy")
            if not os.path.exists(path):
                print(f"Can't find IoU path at {path}, exiting")
                print(f"If this is first run for dataset+method, set use_cached_calculations to False")
                os._exit(1)
            print(f"Using cached IoU at {path}")
            IoU_arr = np.load(path)


        if self.debug:
            print("")
            print("--------------------------------")
            np.set_printoptions(suppress=True, formatter={'float_kind':'{:1.3f}'.format}, linewidth=250)
            print("IoU_arr:")
            print(IoU_arr)
            print("--------------------------------")

        # 2. Do hungarian matching to match gt instances and prediction

        print("")
        print(f"Performing hungarian matching")

        start = timer()

        hungarian_matching = self.hungarian_matching(IoU_arr[:,:len(self.gt_eval)])

        row_ind, col_ind = hungarian_matching

        end = timer()
        print(f"Running time to perform hungarian matching: {end - start:.3f} s") 

        if self.debug:
            print("")
            print("--------------------------------")
            print("Row indices and column indices from hungarian matching:")
            print(row_ind)
            print(col_ind)
            print("--------------------------------")

        # 3. Get TP, FP, FN based on hungarian matching and IoU

        print(f"Classifying predictions based on matches and IoU")

        tp_predictions, tp_gt, fp_predictions, fn_gt, neglected_predictions, tp_names = self.check_matched_predictions(self.predictions, self.gt_eval, IoU_arr, hungarian_matching, self.tree_names)

        if self.debug:
            print("")
            print("--------------------------------")
            print(f"Len of tp_predictions: {len(tp_predictions)}")
            print(f"Len of tp_gt: {len(tp_gt)}")
            print(f"Len of fp_predictions: {len(fp_predictions)}")
            print(f"Len of fn_gt: {len(fn_gt)}")
            print(f"Len of neglected_predictions: {len(neglected_predictions)}")
            print("--------------------------------")

        # 4. Calculate plot-wide metrics

        print(f"Calculating plot wide metrics")

        # get plot-wide metrics
        self.get_plot_wide_metrics(tp_predictions, tp_gt, fp_predictions, fn_gt, odir=odir, print_metrics=self.debug)

        # 5. Calculate tree-level metrics

        print(f"Calculating tree-wide metrics")

        self.get_tree_metrics(tp_predictions, tp_gt, odir=odir, print_metrics=self.debug)

        # 6. Debug: Visualize particular trees or plots

        # self.scatter_height_IoU(self.predictions, self.gt_eval, IoU_arr, hungarian_matching)

        # PLAN: for each succesfull prediction: save point cloud of TP, FP and FN so we can calculate metrics by height
        odir_trees = os.path.join(odir, "results_single_trees")
        if not os.path.exists(odir_trees):
            os.mkdir(odir_trees)
        self.output_matches(tp_predictions, tp_gt, tp_names, odir=odir_trees)

        return


    def calculate_IoU(self, predictions, gt_instances, debug):
        '''
            Calculates IoU for array of predictions and gt_instances.
            Both args should be a list of o3d.t.geometry.PointCloud instances.

            Returns 2d numpy array of IoU where pos (i,j) is IoU of prediction i and gt instance j.
        '''

        iou_matrix = []

        no_overlap_count = 0
        overlap_count = 0
        
        for i, pred in enumerate(predictions):
            if (i+1) % 20 == 0:
                print(f"IoU calculation {i+1} / {len(predictions)}")

            bbox_pred = pred.get_axis_aligned_bounding_box()
            points_pred = np.float32(pred.point.positions.numpy())
            iou_list = []

            for gt in gt_instances:
                # check overlap between bboxs firstfor faster calculation
                bbox_gt = gt.get_axis_aligned_bounding_box()
                overlap = self.check_bbox_overlap(bbox_pred, bbox_gt)

                if overlap:
                    # compute IoU between prediction and tree
                    points_gt = gt.point.positions.numpy()

                    intersect_mask_pred, intersect_mask_gt = self.overlap_points(points_pred, points_gt)

                    intersect_count = np.sum(intersect_mask_gt)
                    union_count = len(points_pred) + len(points_gt) - intersect_count

                    IoU = intersect_count/union_count
                    iou_list.append(IoU)

                    overlap_count += 1
                else:
                    iou_list.append(0.0)
                    no_overlap_count += 1

            iou_matrix.append(iou_list)

        if debug:
            print(f"No overlap count: {no_overlap_count} / {len(predictions)*len(gt_instances)}")
            print(f"Overlap count: {overlap_count} / {len(predictions)*len(gt_instances)}")
        
        return np.array(iou_matrix)
            
    def hungarian_matching(self, IoU_arr):
        '''
            Performes hungarian matching on array of IoU.

            Returns row and column indices of maximized matches.
        '''
        row_ind, col_ind = linear_sum_assignment(IoU_arr, maximize=True)
        return row_ind, col_ind

    def check_matched_predictions(self, predictions, gt_eval_instances, IoU_arr, hungarian_matching, tree_names):

        IOU_THRESHOLD = 0.5

        hung_row_ind, hung_col_ind = hungarian_matching

        true_positive_predictions = []
        true_positive_gt = []
        false_positive_predictions = []
        false_negative_gt = []
        neglected_predictions = []
        tp_names = []

        # loop over all matched gt instances first
        # for each gt instance: get best prediction match:
        # if over threshold: TP prediction, TP GT
        # if under threshold: FN GT, FP prediction if max overlap is with GT tree (neglected if max overlap is with non eval tree
        for i, row_idx in enumerate(hung_row_ind):
            column_idx = hung_col_ind[i]

            IoU = IoU_arr[row_idx][column_idx]

            if IoU > IOU_THRESHOLD:
                true_positive_predictions.append(predictions[row_idx])
                true_positive_gt.append(gt_eval_instances[column_idx])
                tp_names.append(tree_names[column_idx])
            else:
                # check if max overlap is with a eval or non-eval tree (only FP if with eval tree)
                max_IoU = np.max(IoU_arr[row_idx])
                max_IoU_idx = np.argmax(IoU_arr[row_idx])

                if max_IoU == 0.0 or max_IoU_idx in range(len(gt_eval_instances)):
                    # if it also doesn't match well with a non-eval instance, we count it as a false positive
                    false_positive_predictions.append(predictions[row_idx])
                else:
                    neglected_predictions.append(predictions[row_idx])
                # the gt instance is always a FN
                false_negative_gt.append(gt_eval_instances[column_idx])

        # Each gt_eval_instance that is not present in column_idx is also a false negative
        # if number of predictions > number of gt instances, this does nothing
        for i in range(len(gt_eval_instances)):
            if i not in hung_col_ind:
                false_negative_gt.append(gt_eval_instances[i])

        # if number of predictions < number of eval_instances, this loop doesn't do anything
        # check all predictions that were not best match with a GT instance
        # check if their max overlap is with a GT instance
        # if so -> FP
        # if not: if max overlap is 0, probably matches understory, so count as FP, otherwise neglect
        for i, prediction in enumerate(predictions):
            if i in hung_row_ind:
                # already handled in first loop
                continue
            
            max_IoU = np.max(IoU_arr[i])
            max_IoU_idx = np.argmax(IoU_arr[i])

            if max_IoU == 0.0 or max_IoU_idx in range(len(gt_eval_instances)):
                # max IoU with an eval instance (or max IoU is 0.0) -> FP
                false_positive_predictions.append(prediction)
            else:
                # if max IoU with a non-eval instance, we don't count it as a FP
                neglected_predictions.append(prediction)

        return true_positive_predictions, true_positive_gt, false_positive_predictions, false_negative_gt, neglected_predictions, tp_names


    # METRIC CALCULATION

    def get_plot_wide_metrics(self, tp_predictions, tp_gt, fp_predictions, fn_gt, odir=None, print_metrics=False):

        Recall = len(tp_predictions) / (len(tp_predictions) + len(fn_gt))
        Precision = len(tp_predictions) / (len(tp_predictions) + len(fp_predictions))

        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        # IoU: should be calculated in tree-level metrics together with mPrec, mRec, mF1

        if print_metrics:
            print("")
            print("--------------------------------")
            print(f"Plot-wide metrics:")
            print(f"tp_pred: {len(tp_predictions)}, tp_gt: {len(tp_gt)}, fp_pred: {len(fp_predictions)}, fn_gt: {len(fn_gt)}")
            print(f"Detected ground truth trees: {len(tp_gt)} / {len(tp_gt + fn_gt)}")
            print(f"Recall: {Recall:.3f}")
            print(f"Precision: {Precision:.3f}")
            print(f"F1-score: {F1:.3f}")
            print("--------------------------------")

        
        if odir is not None:
            with open(os.path.join(odir, "plot_metrics.txt"), 'w+') as f:
                f.write(f"tp_pred: {len(tp_predictions)}, tp_gt: {len(tp_gt)}, fp_pred: {len(fp_predictions)}, fn_gt: {len(fn_gt)}\n")
                f.write(f"Detected ground truth trees: {len(tp_gt)} / {len(tp_gt + fn_gt)}\n")
                f.write(f"Recall: {Recall:.3f}\n")
                f.write(f"Precision: {Precision:.3f}\n")
                f.write(f"F1-score: {F1:.3f}\n")

        return

    def get_tree_metrics(self, tp_preds, tp_gt, odir=None, print_metrics=False):

        sumprec = 0
        sumrec = 0
        sumF1 = 0
        sumIoU = 0

        for pred, gt in zip(tp_preds, tp_gt):
            points_pred = np.float32(pred.point.positions.numpy())
            points_gt = gt.point.positions.numpy()

            intersect_mask_pred, intersect_mask_gt = self.overlap_points(points_pred, points_gt)

            TP = points_pred[intersect_mask_pred]
            FP = points_pred[~intersect_mask_pred]
            FN = points_gt[~intersect_mask_gt]

            tp = len(TP)
            fp = len(FP)
            fn = len(FN)
            tn = 0

            acc, prec, rec, f1, iou, fp_error_rate, fn_error_rate = self.calc_metrics(tp, fp, tn, fn, print_output=False)

            sumprec += prec
            sumrec += rec
            sumF1 += f1
            sumIoU += iou

        mprec = sumprec/len(tp_preds)
        mrec = sumrec/len(tp_preds)
        mF1 = sumF1/len(tp_preds)
        mIoU = sumIoU/len(tp_preds)

        if print_metrics:
            print("")
            print("--------------------------------")
            print(f"Tree level metrics:")
            print(f"Mean Precision: {mprec:.3f}")
            print(f"Mean Recall: {mrec:.3f}")
            print(f"Mean F1: {mF1:3f}")
            print(f"Mean IoU: {mIoU:3f}")
            print("--------------------------------")

        if odir is not None:
            with open(os.path.join(odir, "tree_metrics.txt"), 'w+') as f:
                f.write(f"Mean Precision: {mprec:.3f}\n")
                f.write(f"Mean Recall: {mrec:.3f}\n")
                f.write(f"Mean F1: {mF1:3f}\n")
                f.write(f"Mean IoU: {mIoU:3f}\n")

        return

    def calc_metrics(self, tp, fp, tn, fn, print_output=False):
        '''
            Calculate metrics and optionally prints out summary

            Args: number of true positive, false positive, true negative and false negative predictions
        '''
        # accuracy
        acc = (tp + tn) / (tp + fp + fn + tn)

        # iou
        if tp == 0 and fp == 0 and fn == 0:
            iou = np.nan
            fp_error_rate = np.nan
            fn_error_rate = np.nan
        else:
            iou = tp / (tp + fp + fn)
            fp_error_rate = fp / (tp + fp + fn)
            fn_error_rate = fn / (tp + fp + fn)

        # rec
        if tp + fn == 0:
            rec = np.nan
        else:
            rec = tp / (tp + fn)

        # prec
        if tp + fp == 0:
            prec = np.nan
        else:
            prec = tp / (tp + fp)

        # f1
        if not np.isnan(prec) and not np.isnan(rec) and not (prec == 0 and rec == 0):
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = np.nan

        if print_output:
            print ("")
            print(f"True positive (gt and pred overlap): {tp}")
            print(f"False positive (points part of pred but not gt): {fp}")
            print(f"False negative (points part of gt but not pred): {fn}")
            print(f"True negative (points not part of gt and not pred): {tn}")
            print(f"IoU tp/(tp+fp+fn): {iou:.3f}")
            print(f"Accuracy (tp+tn)/(tp+tn+fp+fn) (how much of prediction is correct): {acc:.3f}")
            print(f"Recall tp/(tp+fn) (how much of gt instance is detected): {rec:.3f}")
            print(f"Precision tp/(tp+fp) (how much of prediction is actually part of gt): {prec:.3f}")
            print(f"F1 2*prec*rec/(prec+rec): {f1:.3f}")
            print("")
        
        return acc, prec, rec, f1, iou, fp_error_rate, fn_error_rate


    # BELOW: helper functions
    def overlap_points(self, points_pred, points_gt):
        # We check overlap by rounding to 2 decimals, not ideal but works for comparing floats

        # Two magic functions to be able to check the overlap between pointclouds
        def view1D(a, b): # a, b are arrays
            a = np.ascontiguousarray(a)
            b = np.ascontiguousarray(b)
            void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
            return a.view(void_dt).ravel(),  b.view(void_dt).ravel()

        def isin_nd(a,b):
            # a,b are the 3D input arrays to give us "isin-like" functionality across them
            A,B = view1D(a.reshape(a.shape[0],-1),b.reshape(b.shape[0],-1))
            return np.isin(A,B)
        
        points_pred_rounded = np.round(points_pred, decimals=2)
        points_gt_rounded = np.round(points_gt, decimals=2)
        
        intersect_mask_pred = isin_nd(points_pred_rounded, points_gt_rounded) # gives mask where prediction points are present in the ground truth
        intersect_mask_gt = isin_nd(points_gt_rounded, points_pred_rounded) # gives mask where ground truth points are present in prediction

        return intersect_mask_pred, intersect_mask_gt
    
    def check_bbox_overlap(self, bbox_1, bbox_2):
        '''
            Args: o3d.t.geometry.AxisAlignedBoundingBox

            Returns True if bboxs overlap
        '''
        idx_tensor = bbox_1.get_point_indices_within_bounding_box(bbox_2.get_box_points())
        idx_tensor2 = bbox_2.get_point_indices_within_bounding_box(bbox_1.get_box_points())
        return (idx_tensor.num_elements() != 0) or (idx_tensor2.num_elements() != 0)

    def output_matches(self, tp_predictions, tp_gt, tp_names, odir):
        '''
            Outputs pcs of TP, FN, FP points of all matched predictions, so calculations of metrics by height are possible later
        '''
        if len(tp_predictions) != len(tp_gt):
            print("Length of predictions not matching length of gt")
            return
        
        # loop over trees and get TP, FN, FP points
        for i, tree in enumerate(tp_names):
            pred = tp_predictions[i]
            gt = tp_gt[i]
            points_pred = np.float32(pred.point.positions.numpy())
            points_gt = gt.point.positions.numpy()

            intersect_mask_pred, intersect_mask_gt = self.overlap_points(points_pred, points_gt)
            TP = points_pred[intersect_mask_pred]
            FP = points_pred[~intersect_mask_pred]
            FN = points_gt[~intersect_mask_gt]

            TP_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(TP))
            FP_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(FP))
            FN_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(FN))

            TP_pc.paint_uniform_color([0,1,0])
            FP_pc.paint_uniform_color([1,0,0])
            FN_pc.paint_uniform_color([0,0,1])

            odir_tree = os.path.join(odir, tree)
            os.makedirs(odir_tree, exist_ok=True)
            if len(TP) > 0:
                o3d.io.write_point_cloud(os.path.join(odir_tree, "TP.ply"), TP_pc)
            if len(FP) > 0:
                o3d.io.write_point_cloud(os.path.join(odir_tree, "FP.ply"), FP_pc)
            if len(FN) > 0:
                o3d.io.write_point_cloud(os.path.join(odir_tree, "FN.ply"), FN_pc)

            # calculate metrics based on number of tp, fp and fn
            tp = len(TP)
            fp = len(FP)
            fn = len(FN)
            tn = 0
            acc, prec, rec, f1, iou, fp_error_rate, fn_error_rate = self.calc_metrics(tp, fp, tn, fn, print_output=False)
            with open(os.path.join(odir_tree, "metrics.txt"), 'w+') as f:
                    f.write(f"True positive (gt and pred overlap): {tp}\n")
                    f.write(f"False positive (points part of pred but not gt): {fp}\n")
                    f.write(f"False negative (points part of gt but not pred): {fn}\n")
                    f.write(f"True negative (points not part of gt and not pred): {tn}\n")
                    f.write(f"IoU tp/(tp+fp+fn): {iou:.3f}\n")
                    f.write(f"Accuracy (tp+tn)/(tp+tn+fp+fn) (how much of prediction is correct): {acc:.3f}\n")
                    f.write(f"Recall tp/(tp+fn) (how much of gt instance is detected): {rec:.3f}\n")
                    f.write(f"Precision tp/(tp+fp) (how much of prediction is actually part of gt): {prec:.3f}\n")
                    f.write(f"F1 2*prec*rec/(prec+rec): {f1:.3f}\n")
        
        return

    # VISUALIZATION

    def scatter_height_IoU(self, predictions, gt_eval_instances, IoU_arr, hungarian_matching, odir=None):
        print("Building scatterplot of height vs IoU")
        height_array_gt = []
        height_array_predictions = []
        IoU_array = []

        hung_row_ind, hung_col_ind = hungarian_matching

        for i, row_idx in enumerate(hung_row_ind):
            column_idx = hung_col_ind[i]

            IoU = IoU_arr[row_idx][column_idx]
            IoU_array.append(IoU)

            gt_instance = gt_eval_instances[column_idx]
            height_array_gt.append(self.get_height(gt_instance))

            prediction = predictions[row_idx]
            height_array_predictions.append(self.get_height(prediction))


        # this does nothing if the number of predictions is larger then the number of trees
        for i in range(len(gt_eval_instances)):
            if i not in hung_col_ind:
                IoU_array.append(0.0)
                instance = gt_eval_instances[i]
                height_array_gt.append(self.get_height(instance))

        # TODO: TEMP write arrays to file for quick plot development in seperate scripts
        np.save("height_gt.npy", np.array(height_array_gt))
        np.save("height_array_predictions.npy", np.array(height_array_predictions))
        np.save("IoU_array.npy", np.array(IoU_array))

        plt.scatter(height_array_gt, IoU_array, c="green")
        plt.scatter(height_array_predictions, IoU_array[:len(height_array_predictions)], c="red")
        plt.show()
        
        return

    def get_height(self, instance):
        # assumes no outliers
        return instance.get_max_bound().numpy()[2] - instance.get_min_bound().numpy()[2]


if __name__=="__main__":
    Fire(BaseEvaluation)