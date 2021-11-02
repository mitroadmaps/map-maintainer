MapMaintainer: Automatic Updating of Street Maps
================================================

MapMaintainer is a system for updating street maps with newly constructed roads
by analyzing the progression of satellite imagery over time. In contrast to map
extraction methods that process a single satellite image, MapMaintaner compares
old and new images of the same location to detect new roads with high
precision.

This repository contains the MapMaintainer code for "Updating Street Maps using
Changes Detected in Satellite Imagery" (SIGSPATIAL 2021). For more details
about the project, see our project website at https://favyen.com/mapupdate/
where our paper and talk are available.


Setup
-----

tensorflow-gpu 1.15 with Python 3 is needed to run the components.

	pip install tensorflow-gpu==1.15 rtree pillow scipy scikit-image

Go 1.11 is also needed; for newer Go versions, you may need to disable module
support:

	export GO111MODULE=off


Dataset
-------

First, download the Massachusetts dataset. This dataset includes 5,000 square
kilometers of satellite imagery from each of two years in Massachusetts.

	mkdir /data/
	cd /data/
	wget https://favyen.com/files/mass.zip
	unzip mass.zip

The dataset includes satellite imagery from 2013 in `sat-2013/`, imagery from
2018 in `sat-2018/`, and road network graphs in `graphs/`.

Two road networks are included, taken from 2018:
- `graphs/mass.graph` excludes parking and service roads and should be used for
training the tracing model, because we want the model to focus on tracing major
roads.
- `graphs/mass-withall.graph` includes all roads and should be used as the base
map during inference, because during inference we want to make sure we are only
looking for new roads.

The dataset format is detailed at https://github.com/mitroadmaps/roadtracer-beta.
Here is a quick summary:

- `mass_X_Y_sat.jpg` is the satellite image spanning pixel (4096*X, 4096*Y) to
(4096*(X+1), 4096*(Y+1)).
- The road network graph files are in a text format, and each vertex is labeled
with a pixel coordinate (i, j). For example, a vertex at (4200, 1000)
corresponds to pixel (104, 1000) in `mass_1_0_sat.jpg`.

In the commands below, we will assume that the dataset has been extracted in
`/data/`.


Steps
-----

MapMaintainer runs in three steps:

1. Change-seeking iterative tracing: trace newly constructed roads that are
visible in the new satellite image, but neither appear in the current map
dataset nor are visible in the old satellite image.
2. Selective change detection: filter the detected roads from tracing to prune
false positive detections arising from changes in lighting, off-nadir angle,
etc.
3. Post-processing: re-construct a new, updated street map by combining the
existing map with the new road detections.


Change-Seeking Iterative Tracing
--------------------------------

Change-seeking iterative tracing produces an initial set of road detections. We
train a model that extracts a road network from a single satellite image. Then,
we compare model's outputs when independently applied on old and new imagery to
robustly identify newly constructed roads.

Process the road network graphs to create directional labels for training the
iterative tracing model:

	cd changeseeking_tracing
	mkdir /data/angles/
	go run angle_tiles.go mass -20 -10 5 10 /data/graphs/ /data/angles/

This will create labels for tiles spanning from `mass_-20_-10_sat.jpg` to
`mass_5_10_sat.jpg`, which is the training dataset.

Now, we can train the model. It will run for 200 iterations.

	mkdir ./model/
	mkdir ./model/{model_latest,model_best}
	python run_m6.py ./model/ /data/sat-2013/ /data/sat-2018/ /data/graphs/ /data/angles/

Before running inference, create crops of the road network graph corresponding
to each image tile. These will be used as base maps that will be extended
through tracing.

	mkdir /data/tile-graphs-remaining/
	python tile_graphs.py /data/graphs/mass-remaining.graph /data/sat-2018/ /data/tile-graphs-remaining/

Now we can apply the model on the test set, which spans from
`mass_-13_-22_sat.jpg` to `mass_12_-10_sat.jpg`.

	mkdir ./outputs/
	python infer_change.py ./model/model_latest/model /data/sat-2013/ /data/sat-2018/ /data/graphs/ /data/tile-graphs-remaining/ ./outputs/

We can visualize the inferred graph using `vis.go`. For example, to visualize
roads detected in `mass_0_0_sat.jpg`:

	go run vis.go /data/graphs/mass-withall.graph ./outputs/0_0.graph /data/sat-2018/ 0 0

Finally, concatenate all of the output graphs into one road network graph.
Here, we use several confidence thresholds so we can get a precision-recall
tradeoff.

	python graph_cat.py ./outputs/ 30 out_30.graph
	python graph_cat.py ./outputs/ 35 out_35.graph
	python graph_cat.py ./outputs/ 40 out_40.graph
	python graph_cat.py ./outputs/ 45 out_45.graph
	python graph_cat.py ./outputs/ 55 out_55.graph
	python graph_cat.py ./outputs/ 60 out_60.graph
	python graph_cat.py ./outputs/ 65 out_65.graph
	python graph_cat.py ./outputs/ 70 out_70.graph


Selective Change Detection
--------------------------

Selective change detection filters the new roads detected through
change-seeking iterative tracing to prune false positives that arose due to
differences in lightning, off-nadir angle, and other factors between the old
and new satellite image.

Prepare per-tile graphs for training. This time, we use the road network with
parking and service roads excluded:

	cd ../change_model/
	mkdir /data/tile-graphs/
	python prepare_graphs.py /data/graphs/mass.graph /data/sat-2013/ /data/sat-2018/ /data/tile-graphs/

Next, we train the model. This uses a self-supervisory learning signal so that no
labels are required.

	mkdir ./model/
	mkdir ./model/{model_latest,model_best}
	python run.py ./model/ /data/sat-2013/ /data/sat-2018/ /data/tile-graphs/

We can now apply the model on the `.graph` files containing new roads detected
through change-seeking iterative tracing. This will loop through each connected
component, and score that for matching/mismatched classes.

	python apply.py ./model/model_latest/model ../changeseeking_tracing/out_30.graph out_30.graph /data/sat-2013/ /data/sat-2018/

Finally, we can evaluate the output from the second stage:

	cd ..
	go run postprocess/eval.go /data/graphs/mass-pruned.graph change_model/out_30.graph /data/unmatched/
