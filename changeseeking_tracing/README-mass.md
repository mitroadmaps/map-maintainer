Here are instructions for running on the Massachusetts dataset.

First, download the dataset:

	cd changeseeking_tracing
	wget https://favyen.com/files/mass.zip
	unzip mass.zip
	mv sat-2013 imagery-old
	mv sat-2018 imagery-new

There are two graphs.
`graphs/mass.graph` excludes parking and service roads and should be used for training the tracing model,
because we want the model to focus on tracing major roads.
`graphs/mass-withall.graph` includes all roads and should be used as the base map during inference,
because during inference we want to make sure we are only looking for new roads.

Some of the steps below can take a long time and use a lot of memory because the graph files are very large.

We can now compute the angle tiles and train the model:

	mkdir angles
	go run angle_tiles.go mass -13 -22 12 -10
	mkdir model model/model_latest model/model_best
	python run_m6.py

To run inference:

	mkdir tile-graphs
	python tile-graphs.py graphs/mass-withall.graph tile-graphs/
	# Update infer_change.py TILE_LIST to e.g. [(0, 0)]
	python infer_change.py

We can visualize the inferred graph:

	go run vis.go graphs/mass-withall.graph 0_0.graph 0 0

The 0_0.graph and last two arguments should correspond to the tile x,y that we inferred.
