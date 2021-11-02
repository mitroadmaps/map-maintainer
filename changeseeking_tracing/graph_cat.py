from discoverlib import graph, geom
import json
import numpy
import os
import os.path
import sys

in_dir = sys.argv[1]
threshold = int(sys.argv[2])
out_fname = sys.argv[3]

fnames = os.listdir(in_dir)
fnames = [fname for fname in fnames if fname.endswith('.graph')]
g = graph.Graph()
vertex_map = {}
def get_or_create_vertex(x, y):
	if (x, y) not in vertex_map:
		vertex_map[(x, y)] = g.add_vertex(geom.Point(x, y))
	return vertex_map[(x, y)]
for fname in fnames:
	label = fname.split('.')[0]
	subg = graph.read_graph(os.path.join(in_dir, fname), merge_duplicates=True, verbose=False)
	with open(os.path.join(in_dir, fname.replace('.graph', '.json')), 'r') as f:
		edge_probs = json.load(f)

	# only use connected components with avg prob exceeding threshold
	seen = set()
	def dfs(edge, cur):
		if edge.id in seen:
			return
		seen.add(edge.id)
		cur.append(edge.id)
		for other in edge.src.out_edges:
			dfs(other, cur)
		for other in edge.dst.out_edges:
			dfs(other, cur)
	bad = set()
	for edge in subg.edges:
		if edge.id in seen:
			continue
		cur = []
		dfs(edge, cur)
		avg_prob = numpy.mean([edge_probs[edge_id] for edge_id in cur])
		if avg_prob < threshold:
			for edge_id in cur:
				bad.add(edge_id)

	for edge in subg.edges:
		if edge.id in bad:
			continue
		v1 = get_or_create_vertex(edge.src.point.x, edge.src.point.y)
		v2 = get_or_create_vertex(edge.dst.point.x, edge.dst.point.y)
		g.add_bidirectional_edge(v1, v2)

g.save(out_fname)
