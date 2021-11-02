import os
import sys

sys.path.append('../changeseeking_tracing/')
from discoverlib import geom, graph

graph_fname = sys.argv[1]
old_tile_path = sys.argv[2]
new_tile_path = sys.argv[3]
out_path = sys.argv[4]

print('reading graph')
g = graph.read_graph(graph_fname)

print('grid index')
grid_idx = g.edge_grid_index(256)

print('computing subgraphs')
SAT_PATHS = [old_tile_path, new_tile_path]
REGIONS = 'mass'

keys = None
for sat_path in SAT_PATHS:
	path_keys = [fname.split('_sat.jpg')[0] for fname in os.listdir(sat_path) if '_sat.jpg' in fname]
	if keys is None:
		keys = set(path_keys)
	else:
		keys = keys.intersection(path_keys)

keys = [k for k in keys if k.split('_')[0] in REGIONS]
keys = [k for k in keys if int(k.split('_')[1]) >= -20 and int(k.split('_')[1]) < 5]
keys = [k for k in keys if int(k.split('_')[2]) >= -10 and int(k.split('_')[2]) < 10]

for k in keys:
	print('... load {}'.format(k))
	region, x, y = k.split('_')
	x, y = int(x), int(y)

	r = geom.Rectangle(
		geom.Point(x, y).scale(4096),
		geom.Point(x+1, y+1).scale(4096),
	)
	subg = graph.graph_from_edges(grid_idx.search(r))
	for vertex in subg.vertices:
		vertex.point = vertex.point.sub(r.start)
	subg.save(os.path.join(out_path, '{}.graph'.format(k)))
