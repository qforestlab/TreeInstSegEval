# TLS forest instance segmentation benchmark

Repository with code for evaluation of TLS forest instance segmentation methods, as used in the paper "Benchmarking tree instance segmentation of terrestrial laser scanning point clouds".

## Benchmark setup

### Data

Benchmark data can be downloaded here [TODO]

Make sure data is structured as follows:

```
├─ base_dir
    ├─  BASE
        └─  <DATASET>
            ├─  trees
                ├─  test
                    ├─  in_plot_th_0.9
                        ├─  tree_1.ply
                        └─  ...
                    ├─  out_plot_th_0.9
                        ├─  tree_1.ply
                        └─  ...
                ├─  val
                    ├─  tree_1.ply
                    └─  ...
                └─  train
                    ├─  tree_1.ply
                    └─  ...
            ├─  <dataset>_train.ply
            ├─  <dataset>_test.ply
            └─  <dataset>_val.ply
    └─  OUTPUTS
        └─  <method>
            └─  <DATASET>
                └─  output files
```
If evaluating on new data, structure as above and add dataset name to the DATASETS list in base_evaluation.py.

### Code

Main requirements are scipy, open3d and python fire.

Each method has an evaluation class which inherits from the base_evaluation class.
Current included methods are: Rayextract, Treeiso, SSSC, TreeLearn and Xiang et al. (2023)

Evaluating other methods is relatively easy: 
1. Order input and output data using the file structure above 
2. Create a new class that implements the read_output function, which must output a list of open3d.t.geometry.PointCloud instances that represent the model predictions. Point clouds are assumed to be non-ordered, so evaluation is performed in a distance-based manner. Therefore, make sure method predictions are not shifted or scaled in any way. Feel free to open a pull request for any new methods.

Run evaluation using:
```
python <method_evaluation>.py 
--data_base_dir: location of base directory containing input and output data
--dataset: name of dataset
--cache_calculations: if True, will cache IoU calculations
--use_cached_calculations: if True, use cached calculations
--debug: if True, print debug information
```

Evaluation results can be found at `base_dir\EVALUATION\<method>\<DATASET>`.
    - plot_metrics.txt contain plot-level recall, precision and F1-score
    - tree_level_metrics.txt contain tree-level recall, precision, F1 and IoU, averaged over succesfull predictions
    - results_single_trees contains a folder for each succesfully detected tree (IoU > 0.5), with the corresponding TP, FN and FP point clouds and a .txt. with recall, precision, F1 and IoU values


## Cite:

If you use the benchmark data or evaluation code, please cite:

```
TODO
```