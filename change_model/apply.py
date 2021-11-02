import model

import numpy
import random
import skimage.io, skimage.morphology
import sys
import tensorflow as tf

sys.path.append('../changeseeking_tracing/')
from discoverlib import geom, graph

model_path = sys.argv[1]
in_path = sys.argv[2]
out_path = sys.argv[3]
old_tile_path = sys.argv[4]
new_tile_path = sys.argv[5]

MODEL_PATH = model_path
TEST_PATH = in_path
OUT_PATH = out_path
SAT_PATHS = [old_tile_path, new_tile_path]
VIS_PATH = None
SIZE = 1024
THRESHOLD = 0.1

print('initializing model')
m = model.Model(size=SIZE)
session = tf.Session()
m.saver.restore(session, MODEL_PATH)

print('reading inferred graph')
g = graph.read_graph(TEST_PATH)
print('creating edge index')
idx = g.edgeIndex()

# determine which tiles the graph spans
print('identifying spanned tiles')
tiles = set()
for vertex in g.vertices:
	x, y = vertex.point.x/4096, vertex.point.y/4096
	tiles.add(geom.Point(x, y))

counter = 0

out_graph = graph.Graph()
out_vertex_map = {}
def get_out_vertex(p):
	if (p.x, p.y) not in out_vertex_map:
		out_vertex_map[(p.x, p.y)] = out_graph.add_vertex(p)
	return out_vertex_map[(p.x, p.y)]

for tile in tiles:
	print('...', tile)
	tile_rect = geom.Rectangle(
		tile.scale(4096),
		tile.add(geom.Point(1, 1)).scale(4096)
	)
	tile_graph = idx.subgraph(tile_rect)
	for vertex in tile_graph.vertices:
		vertex.point = vertex.point.sub(tile_rect.start)
	if len(tile_graph.edges) == 0:
		continue

	sat1 = skimage.io.imread('{}/mass_{}_{}_sat.jpg'.format(SAT_PATHS[0], tile.x, tile.y))
	sat2 = skimage.io.imread('{}/mass_{}_{}_sat.jpg'.format(SAT_PATHS[1], tile.x, tile.y))
	origin_clip_rect = geom.Rectangle(geom.Point(0, 0), geom.Point(4096-SIZE, 4096-SIZE))

	# loop through connected components, and run our model on each component
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
	for edge in tile_graph.edges:
		if edge.id in seen:
			continue
		cur = []
		dfs(edge, cur)
		cur_edges = [tile_graph.edges[edge_id] for edge_id in cur]

		# Prune small connected components.
		cur_length = sum([edge.segment().length() for edge in cur_edges]) / 2
		if cur_length < 60:
			continue

		subg = graph.graph_from_edges(cur_edges)
		origin = random.choice(subg.vertices).point.sub(geom.Point(SIZE/2, SIZE/2))
		origin = origin_clip_rect.clip(origin)
		im1 = sat1[origin.y:origin.y+SIZE, origin.x:origin.x+SIZE, :]
		im2 = sat2[origin.y:origin.y+SIZE, origin.x:origin.x+SIZE, :]
		im2vis = numpy.copy(im2)
		mask = numpy.zeros((SIZE, SIZE), dtype='bool')
		for edge in subg.edges:
			src = edge.src.point.sub(origin)
			dst = edge.dst.point.sub(origin)
			for p in geom.draw_line(src, dst, geom.Point(SIZE, SIZE)):
				mask[p.y, p.x] = True
				im2vis[p.y, p.x, :] = [255, 255, 0]
		mask = skimage.morphology.binary_dilation(mask, selem=skimage.morphology.disk(15))
		mask = mask.astype('uint8').reshape(SIZE, SIZE, 1)
		mask_tile = numpy.concatenate([mask, mask, mask], axis=2)
		cat_im = numpy.concatenate([im1*mask_tile, im2*mask_tile, mask], axis=2)
		output = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [cat_im],
		})[0]

		if VIS_PATH is not None:
			counter += 1
			skimage.io.imsave('{}/{}-{}-a.jpg'.format(VIS_PATH, counter, output), im1)
			skimage.io.imsave('{}/{}-{}-b.jpg'.format(VIS_PATH, counter, output), im2)
			skimage.io.imsave('{}/{}-{}-bb.jpg'.format(VIS_PATH, counter, output), im2vis)
			skimage.io.imsave('{}/{}-{}-mask-a.jpg'.format(VIS_PATH, counter, output), cat_im[:, :, 0:3])
			skimage.io.imsave('{}/{}-{}-mask-b.jpg'.format(VIS_PATH, counter, output), cat_im[:, :, 3:6])
			skimage.io.imsave('{}/{}-{}-mask.png'.format(VIS_PATH, counter, output), mask[:, :, 0]*255)

		if output < THRESHOLD:
			continue

		# integrate into out_graph
		for edge in subg.edges:
			src = edge.src.point.add(tile_rect.start)
			dst = edge.dst.point.add(tile_rect.start)
			out_graph.add_edge(get_out_vertex(src), get_out_vertex(dst))

out_graph.save(OUT_PATH)
