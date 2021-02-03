from discoverlib import geom, graph, tf_util
import model_m6a as model
import model_utils
import tileloader as tileloader

import json
import numpy
import math
import os.path
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

# you may need to configure these paths
MODEL_PATH = './model/model_latest/model'
tileloader.REGIONS = ['mass']
tileloader.TRAINING_REGIONS = tileloader.REGIONS
tileloader.tile_dir = [
	'./imagery-old/',
	'./imagery-new/',
]
tileloader.graph_dir = './graphs/'
tileloader.angles_dir = './angles/'
tileloader.pytiles_path = './pytiles.json'
tileloader.startlocs_path = './starting_locations.json'
# list of tiles like imagery-{old,new}/mass_0_0.jpg where we should trace roads
TILE_LIST = [(0, 0)]
TILE_GRAPH_PATH = './tile-graphs/'

MAX_PATH_LENGTH = 1000000
SEGMENT_LENGTH = 20
TILE_MODE = 'sat'
THRESHOLD_BRANCH = 0.2
THRESHOLD_FOLLOW = 0.2
WINDOW_SIZE = 256
ANGLE_ONEHOT = 64
M6 = True
CACHE_M6 = True
FILTER_BY_TAG = None

REGION = tileloader.REGIONS[0]
tileloader.tile_size = 4096
tileloader.window_size = 256

vis_counter = 0

def vector_to_action(extension_vertex, angle_outputs, threshold):
	# mask out buckets that are similar to existing edges
	blacklisted_buckets = set()
	for edge in extension_vertex.out_edges:
		angle = geom.Point(1, 0).signed_angle(edge.segment().vector())
		bucket = int((angle + math.pi) * 64.0 / math.pi / 2)
		for offset in xrange(6):
			clockwise_bucket = (bucket + offset) % 64
			counterclockwise_bucket = (bucket + 64 - offset) % 64
			blacklisted_buckets.add(clockwise_bucket)
			blacklisted_buckets.add(counterclockwise_bucket)

	seen_vertices = set()
	search_queue = []
	nearby_points = {}
	seen_vertices.add(extension_vertex)
	for edge in extension_vertex.out_edges:
		search_queue.append((graph.EdgePos(edge, 0), 0))
	while len(search_queue) > 0:
		edge_pos, distance = search_queue[0]
		search_queue = search_queue[1:]
		if distance > 0:
			nearby_points[edge_pos.point()] = distance
		if distance >= 4 * SEGMENT_LENGTH:
			continue

		edge = edge_pos.edge
		l = edge.segment().length()
		if edge_pos.distance + SEGMENT_LENGTH < l:
			search_queue.append((graph.EdgePos(edge, edge_pos.distance + SEGMENT_LENGTH), distance + SEGMENT_LENGTH))
		elif edge.dst not in seen_vertices:
			seen_vertices.add(edge.dst)
			for edge in edge.dst.out_edges:
				search_queue.append((graph.EdgePos(edge, 0), distance + l - edge_pos.distance))

	# any leftover targets above threshold?
	best_bucket = None
	best_value = None
	for bucket in xrange(64):
		if bucket in blacklisted_buckets:
			continue
		next_point = model_utils.get_next_point(extension_vertex.point, bucket, SEGMENT_LENGTH)
		bad = False
		for nearby_point, distance in nearby_points.items():
			if nearby_point.distance(next_point) < 0.5 * (SEGMENT_LENGTH + distance):
				bad = True
				break
		if bad:
			continue

		value = angle_outputs[bucket]
		if value > threshold and (best_bucket is None or value > best_value):
			best_bucket = bucket
			best_value = value

	x = numpy.zeros((64,), dtype='float32')
	if best_bucket is not None:
		x[best_bucket] = best_value
	return x

def eval(paths, m, session, max_path_length=MAX_PATH_LENGTH, segment_length=SEGMENT_LENGTH, save=False, compute_targets=True, max_batch_size=model.BATCH_SIZE, window_size=WINDOW_SIZE, verbose=True, threshold_override=None, cache_m6=None, cache_m6_old=None, cache_m6_new=None):
	global vis_counter
	angle_losses = []
	detect_losses = []
	losses = []
	path_lengths = {path_idx: 0 for path_idx in xrange(len(paths))}

	last_time = None
	big_time = None

	last_extended = False

	for len_it in xrange(99999999):
		if len_it % 500 == 0 and verbose:
			print 'it {}'.format(len_it)
			big_time = time.time()
		path_indices = []
		extension_vertices = []
		for path_idx in xrange(len(paths)):
			if path_lengths[path_idx] >= max_path_length:
				continue
			extension_vertex = paths[path_idx].pop()
			if extension_vertex is None:
				continue
			path_indices.append(path_idx)
			path_lengths[path_idx] += 1
			extension_vertices.append(extension_vertex)

			if len(path_indices) >= max_batch_size:
				break

		if len(path_indices) == 0:
			break

		batch_inputs = []
		batch_detect_targets = []
		batch_angle_targets = numpy.zeros((len(path_indices), 64), 'float32')
		inputs_per_path = 1

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]

			path_input, path_detect_target = model_utils.make_path_input(paths[path_idx], extension_vertices[i], segment_length, window_size=window_size)
			if type(path_input) == list:
				batch_inputs.extend([x[:, :, 0:3] for x in path_input])
				inputs_per_path = len(path_input)
				#batch_inputs.append(numpy.concatenate([x[:, :, 0:3] for x in path_input], axis=2))
			else:
				batch_inputs.append(path_input[:, :, 0:3])
			#batch_detect_targets.append(path_detect_target)
			batch_detect_targets.append(numpy.zeros((64, 64, 1), dtype='float32'))

			if compute_targets:
				angle_targets, _ = model_utils.compute_targets_by_best(paths[path_idx], extension_vertices[i], segment_length)
				batch_angle_targets[i, :] = angle_targets

		# run model
		if M6:
			angle_loss, detect_loss, loss = 0.0, 0.0, 0.0
			if cache_m6 is not None:
				p = extension_vertices[0].point.sub(paths[0].tile_data['rect'].start).scale(0.25)
				batch_angle_outputs = numpy.array([cache_m6[p.x, p.y, :]], dtype='float32')
			else:
				pre_outputs = session.run(m.outputs, feed_dict={
					m.is_training: False,
					m.inputs: batch_inputs,
				})
				batch_angle_outputs = pre_outputs[:, window_size/8, window_size/8, :]
		else:
			feed_dict = {
				m.is_training: False,
				m.inputs: batch_inputs,
				m.angle_targets: [x for x in batch_angle_targets for _ in xrange(inputs_per_path)],
				m.detect_targets: [x for x in batch_detect_targets for _ in xrange(inputs_per_path)],
			}
			if ANGLE_ONEHOT:
				feed_dict[m.angle_onehot] = model_utils.get_angle_onehot(ANGLE_ONEHOT)
			batch_angle_outputs, angle_loss, detect_loss, loss = session.run([m.angle_outputs, m.angle_loss, m.detect_loss, m.loss], feed_dict=feed_dict)

		if inputs_per_path > 1:
			actual_outputs = numpy.zeros((len(path_indices), 64), 'float32')
			for i in xrange(len(path_indices)):
				actual_outputs[i, :] = batch_angle_outputs[i*inputs_per_path:(i+1)*inputs_per_path, :].max(axis=0)
			batch_angle_outputs = actual_outputs

		angle_losses.append(angle_loss)
		losses.append(loss)

		if (save is True and len_it % 1 == 0 and vector_to_action(extension_vertices[0], cache_m6_new[p.x, p.y, :], 0.3).max() > 0) or (save == 'extends' and last_extended):
			fname = '/home/ubuntu/data/{}_'.format(vis_counter)
			vis_counter += 1
			save_angle_targets = batch_angle_targets[0, :]
			if not compute_targets:
				save_angle_targets = None
			#if numpy.max(batch_angle_outputs[0, :]) > 0.1:
			#	batch_angle_outputs[0, :] *= 1.0 / numpy.max(batch_angle_outputs[0, :])
			#model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname, angle_targets=save_angle_targets, angle_outputs=batch_angle_outputs[0, :], window_size=window_size)
			old_outputs = numpy.array([cache_m6_old[p.x, p.y, :]], dtype='float32')
			new_outputs = numpy.array([cache_m6_new[p.x, p.y, :]], dtype='float32')
			model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname + 'old_', angle_targets=save_angle_targets, angle_outputs=old_outputs[0, :], window_size=window_size)
			model_utils.make_path_input(paths[path_indices[0]], extension_vertices[0], segment_length, fname=fname + 'new_', angle_targets=save_angle_targets, angle_outputs=new_outputs[0, :], window_size=window_size)

			with open(fname + 'meta.txt', 'w') as f:
				f.write('max angle output: {}\n'.format(batch_angle_outputs[0, :].max()))

		for i in xrange(len(path_indices)):
			path_idx = path_indices[i]
			if len(extension_vertices[i].out_edges) >= 2:
				threshold = THRESHOLD_BRANCH
			else:
				threshold = THRESHOLD_FOLLOW
			if threshold_override is not None:
				threshold = threshold_override

			x = vector_to_action(extension_vertices[i], batch_angle_outputs[i, :], threshold)
			last_extended = x.max() > 0
			paths[path_idx].push(extension_vertices[i], x, segment_length, training=False, branch_threshold=0.01, follow_threshold=0.01, point_reconnect=False)

	if save:
		paths[0].graph.save('out.graph')

	return numpy.mean(angle_losses), numpy.mean(detect_losses), numpy.mean(losses), len_it

def graph_filter(g, threshold=0.3, min_len=None):
	road_segments, _ = graph.get_graph_road_segments(g)
	bad_edges = set()
	for rs in road_segments:
		if min_len is not None and len(rs.edges) < min_len:
			bad_edges.update(rs.edges)
			continue
		probs = []
		if len(rs.edges) < 5 or True:
			for edge in rs.edges:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		else:
			for edge in rs.edges[2:-2]:
				if hasattr(edge, 'prob'):
					probs.append(edge.prob)
		if not probs:
			continue
		avg_prob = numpy.mean(probs)
		if avg_prob < threshold:
			bad_edges.update(rs.edges)
	print 'filtering {} edges'.format(len(bad_edges))
	ng = graph.Graph()
	vertex_map = {}
	for vertex in g.vertices:
		vertex_map[vertex] = ng.add_vertex(vertex.point)
	for edge in g.edges:
		if edge not in bad_edges:
			ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
	return ng

if __name__ == '__main__':
	print 'reading tiles'
	tiles = tileloader.Tiles(2, SEGMENT_LENGTH, 16, TILE_MODE)

	print 'initializing model'
	m = model.Model(bn=True, size=2048)
	session = tf.Session()
	m.saver.restore(session, MODEL_PATH)

	def too_close(idx, p):
		for edge in idx.search(p.bounds().add_tol(60)):
			if edge.segment().distance(p) < 60:
				return True
		return False

	for x, y in TILE_LIST:
		EXISTING_GRAPH_FNAME = '{}/{}_{}.graph'.format(TILE_GRAPH_PATH, x, y)
		r = geom.Rectangle(
			geom.Point(x*4096, y*4096),
			geom.Point((x+1)*4096, (y+1)*4096)
		)

		g = graph.read_graph(EXISTING_GRAPH_FNAME)
		gc = graph.GraphContainer(g)
		g = g.clone()
		graph.densify(g, SEGMENT_LENGTH)
		tile_data = {
			'region': REGION,
			'rect': r,
			'search_rect': r.add_tol(-WINDOW_SIZE/2),
			'cache': tiles.cache,
			'starting_locations': [],
		}
		#path = model_utils.Path(gc, tile_data, g=g)
		path = model_utils.Path(None, tile_data, g=g)
		for vertex in g.vertices:
			vertex.edge_pos = None
			path.prepend_search_vertex(vertex)

		big_ims = tile_data['cache'].get(tile_data['region'], tile_data['rect'])
		m6_old = tf_util.apply_conv(session, m, big_ims['input1'], scale=4, channels=64)
		m6_new = tf_util.apply_conv(session, m, big_ims['input'], scale=4, channels=64)
		#m6_mask = (m6_old < 0.1).astype('float32')
		m6_mask = m6_old > 0.4
		import skimage.morphology
		m6_mask = m6_mask.reshape((1024*1024, 64))
		m6_mask = numpy.concatenate([m6_mask, m6_mask], axis=1)
		m6_mask = skimage.morphology.binary_dilation(m6_mask, numpy.ones((1, 5), dtype='bool'))
		m6_mask = numpy.concatenate([m6_mask[:, 64:96], m6_mask[:, 32:64]], axis=1).reshape((1024, 1024, 64)).astype('float32')
		m6_mask = 1-m6_mask
		cache_m6 = m6_new * m6_mask

		result = eval([path], m, session, save=False, compute_targets=False, cache_m6=cache_m6, cache_m6_old=m6_old, cache_m6_new=m6_new)

		path.graph.save('{}_{}.graph'.format(x, y))
		edge_probs = []
		for edge in ng.edges:
			edge_probs.append(int(edge.prob * 100))
		with open('{}_{}.json'.format(x, y), 'w') as f:
			json.dump(edge_probs, f)
