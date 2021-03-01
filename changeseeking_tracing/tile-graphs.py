from discoverlib import geom, graph
import os, os.path
import sys

graph_fname = sys.argv[1]
out_dir = sys.argv[2]

print('reading graph')
g = graph.read_graph(graph_fname)
print('creating index')
idx = g.edgeIndex()

fnames = os.listdir('imagery-new')
fnames = [fname.split('_') for fname in fnames if fname.endswith('.jpg')]

# list of tuples (x, y) indicating tiles (like imagery-new/region_x_y.jpg) where we want to crop the graph
tiles = [(int(parts[1]), int(parts[2])) for parts in fnames]

for x, y in tiles:
	r = geom.Rectangle(
		geom.Point(x*4096, y*4096),
		geom.Point((x+1)*4096, (y+1)*4096),
	)
	sg = idx.subgraph(r)
	sg.save(os.path.join(out_dir, '{}_{}.graph'.format(x, y)))
