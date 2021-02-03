Change-Seeking Iterative Tracing
================================

This component applies change-seeking iterative tracing to trace newly
constructed roads that are visible in the new satellite image, but neither
appear in the current map dataset nor are visible in the old satellite image.

Setup
-----

tensorflow-gpu 1.15 is needed to run this code.

The expected data format for satellite images and road network graphs is the
same as for https://github.com/mitroadmaps/roadtracer-beta. There are detailed
instructions here. It should end up in a few folders:

* imagery-old/region_0_0_sat.jpg is the old satellite image from (0, 0) to (4096, 4096)
* imagery-new/region_0_0_sat.jpg is the new satellite image from (0, 0) to (4096, 4096)
* graphs/region.graph is the road network graph

Pre-process the data to create some additional files needed for iterative tracing:

	go run angle_tiles.go region 0 0 1 1

Here, the arguments are the bounding box for the tiles, e.g. if you have four images:

[region_0_0_sat.jpg, region_0_1_sat.jpg, region_1_0_sat.jpg, region_1_1_sat.jpg]

Then you should run:

	go run angle_tiles.go region 0 0 2 2

Training
--------

Once the files are setup, training is straightforward:

	mkdir model model/model_latest model/model_best
	python run_m6.py

Inference
---------

To run inference in some new tile, e.g. region_5_5_sat.jpg:

* Update infer_change.py TILE_LIST to [(5, 5)]
* Run python infer_change.py

It should output a new graph 5_5.graph.
