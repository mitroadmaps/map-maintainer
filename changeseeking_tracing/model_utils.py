from discoverlib import geom, graph

import json
import numpy
import math
from PIL import Image
import random
import rtree
import scipy.ndimage
import sys
import time

DEBUG = False

class Path(object):
	def __init__(self, gc, tile_data, start_loc=None, g=None):
		self.gc = gc
		self.tile_data = tile_data
		if g is None:
			self.graph = graph.Graph()
		else:
			self.graph = g
		self.explored_pairs = {}
		self.unmatched_vertices = 0

		if start_loc:
			#v1 = self.graph.add_vertex(start_loc[0]['point'])
			v2 = self.graph.add_vertex(start_loc[1]['point'])
			#self.graph.add_bidirectional_edge(v1, v2)

			#v1.edge_pos = start_loc[0]['edge_pos']
			v2.edge_pos = start_loc[1]['edge_pos']

			self.search_vertices = [v2]
		else:
			self.search_vertices = []

		self._load_edge_rtree()

	def _load_edge_rtree(self):
		self.indexed_edges = set()
		self.edge_rtree = rtree.index.Index()
		for edge in self.graph.edges:
			self._add_edge_to_rtree(edge)

	def _add_edge_to_rtree(self, edge):
		if edge.id in self.indexed_edges:
			return
		self.indexed_edges.add(edge.id)
		bounds = edge.segment().bounds().add_tol(1)
		self.edge_rtree.insert(edge.id, (bounds.start.x, bounds.start.y, bounds.end.x, bounds.end.y))

	def _add_bidirectional_edge(self, src, dst, prob=1.0):
		edges = self.graph.add_bidirectional_edge(src, dst)
		if prob is not None:
			edges[0].prob = prob
			edges[1].prob = prob
		self._add_edge_to_rtree(edges[0])
		self._add_edge_to_rtree(edges[1])

	def prepend_search_vertex(self, vertex):
		if self.tile_data['search_rect'].contains(vertex.point):
			self.search_vertices = [vertex] + self.search_vertices
			return True
		else:
			return False

	def get_path_to(self, vertex, path=None, limit=6):
		if path is None:
			path = []
		def follow(vertex):
			path.insert(0, vertex)
			if len(path) >= limit:
				return
			for edge in vertex.in_edges:
				if edge.src not in path:
					follow(edge.src)
					return
		follow(vertex)
		return path

	def mark_edge_explored(self, edge, distance):
		l = edge.segment().length()
		if (edge.src.id, edge.dst.id) in self.explored_pairs:
			current_start, current_end = self.explored_pairs[(edge.src.id, edge.dst.id)]
		else:
			current_start, current_end = None, None

		if current_start is None:
			new_start = distance
		else:
			new_start = max(current_start, distance)
		reverse_new_end = l - new_start

		if new_start >= l:
			new_end = -1
			reverse_new_start = l + 1
		elif current_end is None:
			new_end = None
			reverse_new_start = None
		else:
			new_end = current_end
			reverse_new_start = l - current_end

		self.explored_pairs[(edge.src.id, edge.dst.id)] = (new_start, new_end)
		self.explored_pairs[(edge.dst.id, edge.src.id)] = (reverse_new_start, reverse_new_end)

	def mark_rs_explored(self, rs, distance=None):
		for edge in rs.edges:
			edge_distance = rs.edge_distances[edge.id]
			l = edge.segment().length()
			if distance is None or distance >= edge_distance + l:
				self.mark_edge_explored(edge, l + 1)
			elif distance < edge_distance:
				break
			else:
				self.mark_edge_explored(edge, distance - edge_distance)
				break

	def is_explored(self, edge_pos):
		if (edge_pos.edge.src.id, edge_pos.edge.dst.id) not in self.explored_pairs:
			return False
		start, end = self.explored_pairs[(edge_pos.edge.src.id, edge_pos.edge.dst.id)]
		if (start is None or edge_pos.distance >= start) and (end is None or edge_pos.distance <= end):
			return False
		return True

	def push(self, extension_vertex, angle_outputs, segment_length, training=True, branch_threshold=0.2, follow_threshold=0.2, allow_reconnect=True, force_reconnect=False, point_reconnect=False):
		if DEBUG: print 'extending from {}'.format(extension_vertex.point)
		max_angle = numpy.max(angle_outputs)

		if force_reconnect and allow_reconnect and len(extension_vertex.in_edges) >= 1:
			nearby_vertices = graph.get_nearby_vertices_by_distance(extension_vertex, 6*segment_length)
			prev_vertex = extension_vertex.in_edges[0].src

			best_vertex = None
			best_distance = None
			possible_rect = extension_vertex.point.bounds().add_tol(6 * segment_length)
			for edge_id in self.edge_rtree.intersection((possible_rect.start.x, possible_rect.start.y, possible_rect.end.x, possible_rect.end.y)):
				edge = self.graph.edges[edge_id]
				if edge.src in nearby_vertices or edge.dst in nearby_vertices:
					continue
				for vertex in [edge.src, edge.dst]:
					vector_to_vertex = vertex.point.sub(extension_vertex.point)
					vector_to_extension = extension_vertex.point.sub(prev_vertex.point)
					if vector_to_vertex.angle_to(vector_to_extension) > math.pi / 2:
						continue

					distance = vertex.point.distance(extension_vertex.point) + vector_to_vertex.angle_to(vector_to_extension) * segment_length
					if len(vertex.out_edges) >= 2:
						distance -= segment_length / 2
					if best_vertex is None or distance < best_distance:
						best_vertex = vertex
						best_distance = distance
			if best_vertex is not None:
				if DEBUG: print '... push: force reconnect with existing vertex at {}'.format(best_vertex.point)

				if self.gc is not None:
					# mark path up to best_vertex as explored
					path = self.get_path_to(extension_vertex)
					if best_vertex.edge_pos is not None:
						path += reversed(self.get_path_to(best_vertex, limit=3))
					else:
						path.append(best_vertex)

					probs, backpointers = graph.mapmatch(self.gc.edge_index, self.gc.road_segments, self.gc.edge_to_rs, [vertex.point for vertex in path], segment_length)
					if probs is not None:
						best_rs = graph.mm_best_rs(self.gc.road_segments, probs)
						rs_list = graph.mm_follow_backpointers(self.gc.road_segments, best_rs.id, backpointers)
						if DEBUG: print '... push: force reconnect: marking explored rs: {}'.format([rs.id for rs in rs_list[2:]])
						for rs in set(rs_list[2:] + [best_rs]):
							self.mark_rs_explored(rs)

				self._add_bidirectional_edge(extension_vertex, best_vertex)
				return

		if max_angle < follow_threshold or (max_angle < branch_threshold and len(extension_vertex.out_edges) >= 2) or len(extension_vertex.out_edges) > 4:
			if DEBUG: print '... push: decided to stop'

			if self.gc is not None and len(extension_vertex.out_edges) >= 1:
				# stop; we should mark path explored
				path = self.get_path_to(extension_vertex)
				probs, backpointers = graph.mapmatch(self.gc.edge_index, self.gc.road_segments, self.gc.edge_to_rs, [vertex.point for vertex in path], segment_length)
				if probs is not None:
					best_rs = graph.mm_best_rs(self.gc.road_segments, probs)
					rs_list = graph.mm_follow_backpointers(self.gc.road_segments, best_rs.id, backpointers)
					if DEBUG: print '... push: stop, so marking rs explored: ({})'.format([rs.id for rs in rs_list[2:] + [best_rs]])
					for rs in set([rs for rs in rs_list[2:] if rs != best_rs]):
						self.mark_rs_explored(rs)
					best_pos = best_rs.closest_pos(extension_vertex.point)
					self.mark_rs_explored(best_rs, distance=best_rs.edge_distances[best_pos.edge.id]+best_pos.distance)
		else:
			if training and random.random() < 0.1 and False:
				angle_bucket = random.randint(0, 63)
			else:
				angle_bucket = numpy.argmax(angle_outputs)
			angle_prob = angle_outputs[angle_bucket]

			next_point = get_next_point(extension_vertex.point, angle_bucket, segment_length)

			if allow_reconnect:
				# if this point is close to non-nearby vertex, then connect to extension_vertex
				if point_reconnect:
					reconnect_threshold = 1.5 * segment_length
				else:
					reconnect_threshold = 3 * segment_length

				nearby_vertices = graph.get_nearby_vertices_by_distance(extension_vertex, 6*segment_length)

				best_vertex = None
				best_distance = None
				possible_rect = next_point.bounds().add_tol(reconnect_threshold)
				for edge_id in self.edge_rtree.intersection((possible_rect.start.x, possible_rect.start.y, possible_rect.end.x, possible_rect.end.y)):
					edge = self.graph.edges[edge_id]
					if edge.src in nearby_vertices or edge.dst in nearby_vertices:
						continue
					if edge.segment().distance(next_point) > reconnect_threshold:
						continue

					# parallel road constraint: don't reconnect if angle of segments are almost the same
					vector_to_next = next_point.sub(extension_vertex.point)
					edge_vector = edge.segment().vector()
					if not point_reconnect and len(edge.dst.out_edges) >= 2 and (vector_to_next.angle_to(edge_vector) < math.pi / 10 or vector_to_next.angle_to(edge_vector) > math.pi * 9 / 10):
						continue

					proj_point = edge.segment().project(next_point)
					vector_to_point = proj_point.sub(extension_vertex.point)
					if vector_to_point.angle_to(vector_to_next) > math.pi / 4:
						continue

					distance = proj_point.distance(next_point)
					if best_vertex is None or distance < best_distance:
						best_vertex = (edge, proj_point)
						best_distance = distance

					'''for vertex in [edge.src, edge.dst]:
						vector_to_vertex = vertex.point.sub(extension_vertex.point)
						if vector_to_vertex.angle_to(vector_to_next) > math.pi / 4:
							continue

						distance = vertex.point.distance(next_point)
						if len(vertex.out_edges) >= 2:
							distance -= segment_length / 2
						if best_vertex is None or distance < best_distance:
							best_vertex = vertex
							best_distance = distance'''
				if best_vertex is not None:
					edge, proj_point = best_vertex
					if DEBUG: print '... push: decided to reconnect with existing vertex at {}'.format(proj_point)
					new_vertex = self.graph.add_vertex(proj_point)
					if hasattr(edge.src, 'edge_pos'):
						new_vertex.edge_pos = edge.src.edge_pos
					#new_vertex = best_vertex

					if self.gc is not None:
						# mark path up to best_vertex as explored
						path = self.get_path_to(extension_vertex)
						if new_vertex.edge_pos is not None:
							path += reversed(self.get_path_to(new_vertex, limit=3))
						else:
							path.append(new_vertex)
						probs, backpointers = graph.mapmatch(self.gc.edge_index, self.gc.road_segments, self.gc.edge_to_rs, [vertex.point for vertex in path], segment_length)
						if probs is not None:
							best_rs = graph.mm_best_rs(self.gc.road_segments, probs)
							rs_list = graph.mm_follow_backpointers(self.gc.road_segments, best_rs.id, backpointers)
							if DEBUG: print '... push: reconnect: marking explored rs: {}'.format([rs.id for rs in rs_list[2:]])
							for rs in set(rs_list[2:] + [best_rs]):
								self.mark_rs_explored(rs)

					# split edge
					src, dst = edge.src, edge.dst
					opp_edge = edge.get_opposite_edge()
					edge.dst = new_vertex
					opp_edge.src = new_vertex
					dst.in_edges.remove(edge)
					dst.out_edges.remove(opp_edge)
					new_vertex.in_edges.append(edge)
					new_vertex.out_edges.append(opp_edge)
					if hasattr(edge, 'prob'):
						self._add_bidirectional_edge(new_vertex, dst, prob=edge.prob)
					else:
						self._add_bidirectional_edge(new_vertex, dst, prob=None)
					self._add_bidirectional_edge(extension_vertex, new_vertex, prob=angle_prob)
					return

			# add vertex and map-match to find edge_pos
			next_vertex = self.graph.add_vertex(next_point)
			self._add_bidirectional_edge(extension_vertex, next_vertex, prob=angle_prob)
			next_vertex.edge_pos = None

			self.prepend_search_vertex(extension_vertex)
			in_bounds = self.prepend_search_vertex(next_vertex)

			if self.gc is not None:
				path_to_next = self.get_path_to(next_vertex)
				probs, backpointers = graph.mapmatch(self.gc.edge_index, self.gc.road_segments, self.gc.edge_to_rs, [vertex.point for vertex in path_to_next], segment_length)
				if probs is not None:
					if DEBUG: print '... push: mm probs: {}'.format(probs)
					best_rs = graph.mm_best_rs(self.gc.road_segments, probs)
					best_pos = best_rs.closest_pos(next_vertex.point)

					# only use best_rs if it is either not explored, or same as previous rs
					if best_rs is not None and (not self.is_explored(best_pos) or (extension_vertex.edge_pos is not None and self.gc.edge_to_rs[extension_vertex.edge_pos.edge.id] == best_rs)):
						next_vertex.edge_pos = best_pos
						if len(path_to_next) >= 10:
							rs_list = graph.mm_follow_backpointers(self.gc.road_segments, best_rs.id, backpointers)
							if DEBUG: print '... push: mm: {}'.format([rs.id for rs in rs_list])
							if in_bounds:
								if DEBUG: print '... push: normal extend, marking explored rs: {}'.format([rs.id for rs in rs_list[2:5] if rs not in rs_list[5:]])
								for rs in rs_list[2:4]:
									if rs in rs_list[4:]:
										# don't mark edges along rs that we might still be following as explored
										continue
									self.mark_rs_explored(rs)
							else:
								if DEBUG: print '... push: normal extend but out of bounds, marking explored rs: {}'.format([rs.id for rs in rs_list[2:] + [best_rs]])
								for rs in set(rs_list[2:] + [best_rs]):
									self.mark_rs_explored(rs)
					else:
						self.unmatched_vertices += 1

	def pop(self):
		if len(self.search_vertices) == 0:
			return None
		vertex = self.search_vertices[0]
		self.search_vertices = self.search_vertices[1:]
		return vertex

	def clone(self):
		other = Path(self.gc, self.tile_data, g=self.graph.clone())
		other.explored_pairs = dict(self.explored_pairs)
		other.unmatched_vertices = self.unmatched_vertices
		other.search_vertices = list(self.search_vertices)
		return other

def get_unexplored_graph(path, extension_vertex, origin, segment_length, window_size):
	tile = numpy.zeros((128, 128), dtype='float32')

	if extension_vertex.edge_pos is None:
		return tile

	def draw(rs, distance, remaining):
		start_edge_idx = rs.distance_to_edge(distance, return_idx=True)
		for edge_idx in xrange(start_edge_idx, len(rs.edges)):
			edge = rs.edges[edge_idx]
			edge_distance = distance - rs.edge_distances[edge.id]

			start_edge_pos = graph.EdgePos(edge, edge_distance)
			start = start_edge_pos.point()

			if edge_distance + remaining < edge.segment().length():
				end_edge_pos = graph.EdgePos(edge, edge_distance + remaining)
				end = end_edge_pos.point()
			else:
				end = edge.dst.point

			for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
				p_small = p.scale(128.0 / window_size)
				tile[p_small.x, p_small.y] = 1.0

			remaining -= edge.segment().length() - edge_distance
			distance = rs.edge_distances[edge.id] + edge.segment().length() + 0.001
			if remaining <= 0:
				return

		for next_rs in rs.out_rs(path.gc.edge_to_rs):
			if rs == next_rs or next_rs.is_opposite(rs):
				continue
			elif path.is_explored(rs.edges[0]):
				continue
			draw(next_rs, 0, remaining)

	cur_edge = extension_vertex.edge_pos.edge
	cur_rs = path.gc.edge_to_rs[cur_edge.id]
	cur_distance = cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance
	prev_rs = None

	if len(extension_vertex.in_edges) >= 1:
		prev_vertex = extension_vertex.in_edges[0].src
		if prev_vertex.edge_pos is not None:
			prev_edge = prev_vertex.edge_pos.edge
			prev_rs = path.gc.edge_to_rs[prev_edge.id]

	potential_rs = []
	if cur_distance + segment_length < cur_rs.length():
		potential_rs.append((cur_rs, 0, False))
	else:
		distance_to_potential = cur_rs.length() - cur_distance
		for rs in cur_rs.out_rs(path.gc.edge_to_rs):
			if rs == cur_rs or rs.is_opposite(cur_rs):
				continue
			potential_rs.append((rs, distance_to_potential, False))
	if cur_distance < segment_length / 2 and prev_rs is not None:
		distance_to_potential = cur_distance
		for rs in cur_rs.in_rs(path.gc.edge_to_rs):
			if rs == cur_rs or rs.is_opposite(cur_rs) or rs == prev_rs or rs.is_opposite(prev_rs):
				continue

			# add the opposite of this rs so that we are going away from extension_vertex
			opposite_rs = path.gc.edge_to_rs[rs.edges[0].get_opposite_edge().id]
			potential_rs.append((opposite_rs, distance_to_potential, True))

	if len(potential_rs) + 1 > len(extension_vertex.out_edges):
		potential_rs = [(rs, d, is_behind) for rs, d, is_behind in potential_rs if not path.is_explored(rs.edges[0])]
		if len(potential_rs) > 0:
			if path.is_explored(cur_rs.edges[0]):
				for rs, d, is_behind in potential_rs:
					draw(rs, 0, segment_length * 5 - d)
			else:
				draw(cur_rs, cur_distance, segment_length * 5)
				for rs, d, is_behind in potential_rs:
					if is_behind:
						draw(rs, 0, segment_length * 5 - d)

	return tile

def get_unexplored_graph2(path, extension_vertex, origin, segment_length, window_size):
	tile = numpy.zeros((128, 128), dtype='float32')

	if extension_vertex.edge_pos is None:
		return tile

	# find edges and partial edge explored by most recent part of path
	recent_explored = set()
	recent_vertices = path.get_path_to(extension_vertex)
	probs, backpointers = graph.mapmatch(path.gc.edge_index, path.gc.road_segments, path.gc.edge_to_rs, [vertex.point for vertex in recent_vertices], segment_length)
	if probs is not None:
		best_rs = graph.mm_best_rs(path.gc.road_segments, probs)
		rs_list = graph.mm_follow_backpointers(path.gc.road_segments, best_rs.id, backpointers)
		for rs in rs_list[2:] + [best_rs]:
			for edge in rs.edges:
				self.explored_pairs.add((edge.src.id, edge.dst.id))

	def draw(rs, distance, remaining, is_explored=False):
		start_edge_idx = rs.distance_to_edge(distance, return_idx=True)
		for edge_idx in xrange(start_edge_idx, len(rs.edges)):
			edge = rs.edges[edge_idx]
			edge_distance = distance - rs.edge_distances[edge.id]

			start_edge_pos = graph.EdgePos(edge, edge_distance)
			start = start_edge_pos.point()

			if edge_distance + remaining < edge.segment().length():
				end_edge_pos = graph.EdgePos(edge, edge_distance + remaining)
				end = end_edge_pos.point()
			else:
				end = edge.dst.point

			if not is_explored:
				for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
					p_small = p.scale(128.0 / window_size)
					tile[p_small.x, p_small.y] = 1.0

			remaining -= edge.segment().length() - edge_distance
			distance = rs.edge_distances[edge.id] + edge.segment().length() + 0.001
			if remaining <= 0:
				return

		for next_rs in rs.out_rs(path.gc.edge_to_rs):
			if rs == next_rs or next_rs.is_opposite(rs):
				continue
			draw(next_rs, 0, remaining, is_explored=path.is_explored(next_rs.edges[0]))

	cur_edge = extension_vertex.edge_pos.edge
	cur_rs = path.gc.edge_to_rs[cur_edge.id]
	cur_distance = cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance
	draw(cur_rs, cur_distance, segment_length * 5, is_explored=path.is_explored(cur_rs.edges[0]))

	opposite_rs = cur_rs.get_opposite_rs(path.gc.edge_to_rs)
	opposite_pos = opposite_rs.closest_pos(extension_vertex.point)
	opposite_distance = opposite_rs.edge_distances[opposite_pos.edge.id] + opposite_pos.distance
	draw(opposite_rs, opposite_distance, segment_length * 5, is_explored=path.is_explored(cur_rs.edges[0]))

	return tile

def get_compact_path_input(path, extension_vertex, window_size=512):
	origin = extension_vertex.point.sub(geom.Point(window_size/2, window_size/2))
	rect = origin.bounds().extend(origin.add(geom.Point(window_size, window_size)))

	path_segments = []
	for edge_id in path.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
		path_segments.append(path.graph.edges[edge_id].segment())

	gt_segments = []
	for edge in path.gc.edge_index.search(rect):
		gt_segments.append(edge.segment())

	return {
		'window_size': window_size,
		'cache': path.tile_data['cache'],
		'region': path.tile_data['region'],
		'big_rect': path.tile_data['rect'],
		'origin': origin,
		'path_segments': path_segments,
		'gt_segments': gt_segments,
	}

static_angle_onehots = {}
def get_angle_onehot(size):
	if size in static_angle_onehots:
		return static_angle_onehots[size]

	x = numpy.zeros((size, size, 64), dtype='float32')
	for i in xrange(size):
		for j in xrange(size):
			di = i - size/2
			dj = j - size/2
			a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * 64 / 2 / math.pi)
			if a >= 64:
				a = 63
			elif a < 0:
				a = 0
			x[i, j, a] = 1
	static_angle_onehots[size] = x
	return x

def uncompact_path_input(d):
	big_origin = d['big_rect'].start
	big_ims = d['cache'].get(d['region'], d['big_rect'])
	tile_origin = d['origin'].sub(big_origin)
	tile_big = big_ims['input'][tile_origin.x:tile_origin.x+d['window_size'], tile_origin.y:tile_origin.y+d['window_size'], :].astype('float32') / 255.0

	tile_path = numpy.zeros((d['window_size'], d['window_size']), dtype='float32')
	for segment in d['path_segments']:
		for p in geom.draw_line(segment.start.sub(d['origin']), segment.end.sub(d['origin']), geom.Point(d['window_size'], d['window_size'])):
			tile_path[p.x, p.y] = 1.0

	tile_graph = numpy.zeros((d['window_size'], d['window_size']), dtype='float32')
	tile_graph_small = numpy.zeros((d['window_size']/4, d['window_size']/4), dtype='float32')
	for segment in d['gt_segments']:
		for p in geom.draw_line(segment.start.sub(d['origin']), segment.end.sub(d['origin']), geom.Point(d['window_size'], d['window_size'])):
			tile_graph[p.x, p.y] = 1.0

			#p_small = p.scale(128.0 / d['window_size'])
			p_small = p.scale(0.25)
			tile_graph_small[p_small.x, p_small.y] = 1.0

	tile_point = numpy.zeros((d['window_size'], d['window_size']), dtype='float32')
	#tile_point[d['window_size']/2, d['window_size']/2] = 1.0

	input = numpy.concatenate([tile_big, tile_path.reshape(d['window_size'], d['window_size'], 1), tile_point.reshape(d['window_size'], d['window_size'], 1)], axis=2)
	detect_target = tile_graph_small
	return input, detect_target.reshape(d['window_size']/4, d['window_size']/4, 1)

def make_path_input(path, extension_vertex, segment_length, fname=None, green_points=None, blue_points=None, angle_outputs=None, angle_targets=None, action_outputs=None, action_targets=None, detect_output=None, detect_mode='normal', window_size=512):
	big_origin = path.tile_data['rect'].start
	big_ims = path.tile_data['cache'].get(path.tile_data['region'], path.tile_data['rect'])

	if not path.tile_data['rect'].add_tol(-window_size/2).contains(extension_vertex.point):
		raise Exception('bad path {}'.format(path))
	origin = extension_vertex.point.sub(geom.Point(window_size/2, window_size/2))
	tile_origin = origin.sub(big_origin)
	rect = origin.bounds().extend(origin.add(geom.Point(window_size, window_size)))

	tile_path = numpy.zeros((window_size, window_size), dtype='float32')
	for edge_id in path.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
		edge = path.graph.edges[edge_id]
		start = edge.src.point
		end = edge.dst.point
		for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
			tile_path[p.x, p.y] = 1.0
	#draw_segments = []
	#for edge_id in path.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
	#	draw_segments.append(path.graph.edges[edge_id].segment())
	#tile_path = geom.draw_lines(draw_segments, shape=(window_size, window_size)).astype('float32')

	tile_point = numpy.zeros((window_size, window_size), dtype='float32')
	#tile_point[window_size/2, window_size/2] = 1.0

	tile_graph = numpy.zeros((window_size, window_size), dtype='float32')
	tile_graph_small = numpy.zeros((window_size/4, window_size/4), dtype='float32')
	if path.gc is not None:
		for edge in path.gc.edge_index.search(rect):
			start = edge.src.point
			end = edge.dst.point
			for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
				tile_graph[p.x, p.y] = 1.0

				#p_small = p.scale(128.0 / window_size)
				p_small = p.scale(0.25)
				tile_graph_small[p_small.x, p_small.y] = 1.0

	#	draw_segments = []
	#	draw_segments_sm = []
	#	for edge in path.gc.edge_index.search(rect):
	#		start = edge.src.point
	#		end = edge.dst.point
	#		draw_segments.append(geom.Segment(start, end))
	#		draw_segments_sm.append(geom.Segment(start.scale(0.25), end.scale(0.25)))
	#	tile_graph = geom.draw_lines(draw_segments, shape=(window_size, window_size)).astype('float32')
	#	tile_graph_small = geom.draw_lines(draw_segments_sm, shape=(window_size/4, window_size/4)).astype('float32')

	inputs = []
	big_inputs = [v for k, v in big_ims.items() if k.startswith('input')]
	for big_input in big_inputs:
		tile_big = big_input[tile_origin.x:tile_origin.x+window_size, tile_origin.y:tile_origin.y+window_size, :].astype('float32') / 255.0
		input_el = numpy.concatenate([tile_big, tile_path.reshape(window_size, window_size, 1), tile_point.reshape(window_size, window_size, 1)], axis=2)
		#input_el = tile_big # REMOVE ME
		inputs.append(input_el)
	if len(inputs) == 1:
		input = inputs[0]
	else:
		input = inputs

	if detect_mode == 'normal':
		detect_target = tile_graph_small
	elif detect_mode == 'unexplored':
		detect_target = get_unexplored_graph(path, extension_vertex, origin, segment_length, window_size)
	elif detect_mode == 'unexplored2':
		detect_target = get_unexplored_graph2(path, extension_vertex, origin, segment_length, window_size)
	else:
		raise Exception('unknown detect mode {}'.format(detect_mode))

	if fname is not None:
		if False:
			Image.fromarray(numpy.swapaxes((tile_big[:, :, 0:3] * 255.0).astype('uint8'), 0, 1)).save(fname + 'sat.png')

			x = numpy.zeros((window_size, window_size, 3), dtype='float32')
			x[:, :, 0] = tile_path

			if green_points:
				for p in green_points:
					p = p.sub(origin)
					x[p.x-2:p.x+2, p.y-2:p.y+2, 1] = 1.0
			if blue_points:
				for p in blue_points:
					p = p.sub(origin)
					x[p.x-2:p.x+2, p.y-2:p.y+2, 2] = 1.0
			if angle_outputs is not None or angle_targets is not None:
				for i in xrange(window_size):
					for j in xrange(window_size):
						di = i - window_size/2
						dj = j - window_size/2
						d = math.sqrt(di * di + dj * dj)
						a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * 64 / 2 / math.pi)
						if a >= 64:
							a = 63
						elif a < 0:
							a = 0
						if d > 100 and d <= 120 and angle_outputs is not None:
							x[i, j, 1] = angle_outputs[a]
						elif d > 140 and d <= 160 and angle_targets is not None:
							x[i, j, 1] = angle_targets[a]

			if path.gc is not None:
				for edge in path.gc.edge_index.search(rect):
					start = edge.src.point
					end = edge.dst.point
					for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
						x[p.x, p.y, 2] = 1.0
						if path.is_explored(edge):
							x[p.x, p.y, 1] = 1.0

			x[window_size/2-3:window_size/2+3, window_size/2-3:window_size/2+3, 1] = 1.0

			Image.fromarray(numpy.swapaxes((x * 255.0).astype('uint8'), 0, 1)).save(fname + 'path.png')
		if True:
			if detect_output is not None:
				x = numpy.zeros((64, 64, 3), dtype='float32')
				threshold = 0.1
				x[:, :, 1] = numpy.logical_and(detect_target > threshold, detect_output > threshold).astype('float32')
				x[:, :, 0] = numpy.logical_and(detect_target <= threshold, detect_output > threshold).astype('float32')
				x[:, :, 2] = numpy.logical_and(detect_target > threshold, detect_output <= threshold).astype('float32')
				Image.fromarray(numpy.swapaxes((x * 255.0).astype('uint8'), 0, 1)).save(fname + 'detect.png')
		if True:
			x = numpy.zeros((window_size, window_size, 3), dtype='float32')
			x[:, :, 0:3] = tile_big[:, :, 0:3]

			for edge_id in path.edge_rtree.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y)):
				edge = path.graph.edges[edge_id]
				start = edge.src.point
				end = edge.dst.point
				for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
					x[p.x, p.y, 0] = 1.0
					x[p.x, p.y, 1] = 0.0
					x[p.x, p.y, 2] = 0.0

			for edge in path.gc.edge_index.search(rect):
				start = edge.src.point
				end = edge.dst.point
				for p in geom.draw_line(start.sub(origin), end.sub(origin), geom.Point(window_size, window_size)):
					x[p.x, p.y, 0] = 0.0
					x[p.x, p.y, 1] = 1.0
					x[p.x, p.y, 0] = 0.0

			if angle_outputs is not None or angle_targets is not None:
				for i in xrange(window_size):
					for j in xrange(window_size):
						di = i - window_size/2
						dj = j - window_size/2
						d = math.sqrt(di * di + dj * dj)
						a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * 64 / 2 / math.pi)
						if a >= 64:
							a = 63
						elif a < 0:
							a = 0
						if d > 100 and d <= 120 and angle_outputs is not None:
							x[i, j, 0] = angle_outputs[a]
							x[i, j, 1] = angle_outputs[a]
							x[i, j, 2] = 0
						elif d > 140 and d <= 160 and angle_targets is not None:
							x[i, j, 0] = angle_targets[a]
							x[i, j, 1] = angle_targets[a]
							x[i, j, 2] = 0

			x[window_size/2-3:window_size/2+3, window_size/2-3:window_size/2+3, 2] = 1.0
			x[window_size/2-3:window_size/2+3, window_size/2-3:window_size/2+3, 0:2] = 0

			viz_points = helper_compute_viz_points(path, extension_vertex, segment_length)
			if viz_points is not None:
				pp = viz_points['mm'].sub(origin)
				x[pp.x-3:pp.x+3, pp.y-3:pp.y+3, 1:3] = 1.0
				for p in viz_points['nx']:
					pp = p.sub(origin)
					x[pp.x-3:pp.x+3, pp.y-3:pp.y+3, 0:3] = 1.0

			Image.fromarray(numpy.swapaxes((x * 255.0).astype('uint8'), 0, 1)).save(fname + 'overlay.png')

	return input, detect_target.reshape(window_size/4, window_size/4, 1)

def vector_from_angle(angle, scale=100):
	return geom.Point(math.cos(angle) * scale, math.sin(angle) * scale)

#cached_bucket_vectors = {}
#for bucket in xrange(64):
#	angle = bucket * math.pi * 2 / 64.0 - math.pi
#	cached_bucket_vectors[(bucket, 20)] = vector_from_angle(angle, 20)

def get_next_point(prev_point, angle_bucket, segment_length):
#	if segment_length == 20:
#		return prev_point.add(cached_bucket_vectors[(angle_bucket, segment_length)])
	angle = angle_bucket * math.pi * 2 / 64.0 - math.pi
	vector = vector_from_angle(angle, segment_length)
	return prev_point.add(vector)

def compute_targets_by_best(path, extension_vertex, segment_length):
	angle_targets = numpy.zeros((64,), 'float32')
	allow_reconnect = False

	def best_angle_to_pos(pos):
		angle_points = [get_next_point(extension_vertex.point, angle_bucket, segment_length) for angle_bucket in xrange(64)]
		distances = [angle_point.distance(pos.point()) for angle_point in angle_points]
		point_angle = numpy.argmin(distances) * math.pi * 2 / 64.0 - math.pi
		edge_angle = geom.Point(1, 0).signed_angle(pos.edge.segment().vector())
		avg_vector = vector_from_angle(point_angle).add(vector_from_angle(edge_angle))
		avg_angle = geom.Point(1, 0).signed_angle(avg_vector)
		return int((avg_angle + math.pi) * 64.0 / math.pi / 2)

	def set_angle_bucket_soft(target_bucket):
		for offset in xrange(31):
			clockwise_bucket = (target_bucket + offset) % 64
			counterclockwise_bucket = (target_bucket + 64 - offset) % 64
			for bucket in [clockwise_bucket, counterclockwise_bucket]:
				angle_targets[bucket] = max(angle_targets[bucket], pow(0.75, offset))

	def set_by_positions(positions):
		# get existing angle buckets, don't use any that are within 3 buckets
		bad_buckets = set()
		for edge in extension_vertex.out_edges:
			edge_angle = geom.Point(1, 0).signed_angle(edge.segment().vector())
			edge_bucket = int((edge_angle + math.pi) * 64.0 / math.pi / 2)
			for offset in xrange(3):
				clockwise_bucket = (edge_bucket + offset) % 64
				counterclockwise_bucket = (edge_bucket + 64 - offset) % 64
				bad_buckets.add(clockwise_bucket)
				bad_buckets.add(counterclockwise_bucket)

		for pos in positions:
			best_angle_bucket = best_angle_to_pos(pos)
			if best_angle_bucket in bad_buckets:
				continue
			set_angle_bucket_soft(best_angle_bucket)

	if extension_vertex.edge_pos is not None:
		cur_edge = extension_vertex.edge_pos.edge
		cur_rs = path.gc.edge_to_rs[cur_edge.id]
		prev_rs = None

		if len(extension_vertex.in_edges) >= 1:
			prev_vertex = extension_vertex.in_edges[0].src
			if prev_vertex.edge_pos is not None:
				prev_edge = prev_vertex.edge_pos.edge
				prev_rs = path.gc.edge_to_rs[prev_edge.id]

		def get_potential_rs(segment_length, allow_backwards):
			potential_rs = []
			if cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance + segment_length < cur_rs.length():
				potential_rs.append(cur_rs)
			else:
				for rs in cur_rs.out_rs(path.gc.edge_to_rs):
					if rs == cur_rs or rs.is_opposite(cur_rs):
						continue
					potential_rs.append(rs)
			if allow_backwards and cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance < segment_length / 2 and prev_rs is not None:
				for rs in cur_rs.in_rs(path.gc.edge_to_rs):
					if rs == cur_rs or rs.is_opposite(cur_rs) or rs == prev_rs or rs.is_opposite(prev_rs):
						continue

					# add the opposite of this rs so that we are going away from extension_vertex
					opposite_rs = path.gc.edge_to_rs[rs.edges[0].get_opposite_edge().id]
					potential_rs.append(opposite_rs)

			# at very beginning of path, we can go in either direction
			if len(path.graph.edges) == 0:
				# TODO: fix get_opposite_rs for loops
				# currently, if there is a loop, then the rs corresponding to the loop may start at
				#   any point along the loop, and get_opposite_rs will fail
				# I think it may be okay if the loop isn't completely isolated (circle with no
				#   intersections), but definitely it fails for isolated loops
				#potential_rs.append(cur_rs.get_opposite_rs(path.gc.edge_to_rs))
				opposite_rs1 = cur_rs.get_opposite_rs(path.gc.edge_to_rs)
				opposite_rs2 = path.gc.edge_to_rs[cur_rs.edges[-1].get_opposite_edge().id]
				potential_rs.append(opposite_rs2)
				if opposite_rs1 != opposite_rs2:
					if opposite_rs1 is None:
						print 'warning: using opposite_rs2 for rs {}'.format(opposite_rs2.id)
					else:
						raise Exception('opposite_rs1 ({}) != opposite_rs2 ({})'.format(opposite_rs1.id, opposite_rs2.id))

			return potential_rs

		potential_rs = get_potential_rs(segment_length, True)
		potential_far_rs = get_potential_rs(3 * segment_length, False)

		# reconnect if there is a nearby path vertex such that:
		# (1) extension_vertex and the nearby vertex are far in the path graph
		# (2) but close in the ground truth graph
		# (3) and rs is either same as current one, or outgoing from a ground truth vertex that we are entering

		if len(extension_vertex.in_edges) <= 1:
			# first, get outgoing road segments from potential_far_rs
			reconnectable_rs = set([rs for rs in potential_far_rs if path.is_explored(graph.EdgePos(rs.edges[0], 0)) and rs != cur_rs and not cur_rs.is_opposite(rs)])
			reconnectable_rs.add(cur_rs)
			reconnectable_rs.add(path.gc.edge_to_rs[cur_rs.edges[-1].get_opposite_edge().id])

			# now, satisfy 1 and 2, while using set above to satisfy 3
			nearby_path_vertices = set(graph.get_nearby_vertices(extension_vertex, 6))
			gt_distances = graph.shortest_distances_from_source(cur_edge.dst, max_distance=3*segment_length)
			r = extension_vertex.point.bounds().add_tol(3*segment_length)
			for nearby_edge_id in path.edge_rtree.intersection((r.start.x, r.start.y, r.end.x, r.end.y)):
				nearby_edge = path.graph.edges[nearby_edge_id]
				for nearby_vertex in [nearby_edge.src, nearby_edge.dst]:
					if nearby_vertex.edge_pos is None:
						continue
					elif nearby_vertex.point.distance(extension_vertex.point) > 3*segment_length:
						continue
					elif nearby_vertex in nearby_path_vertices:
						continue
					elif path.gc.edge_to_rs[nearby_vertex.edge_pos.edge.id] not in reconnectable_rs:
						continue
					gt_distance = min(
						gt_distances.get(nearby_vertex.edge_pos.edge.src.id, 99999) + nearby_vertex.edge_pos.distance,
						gt_distances.get(nearby_vertex.edge_pos.edge.dst.id, 99999) + nearby_vertex.edge_pos.reverse().distance
					)
					if gt_distance < 3*segment_length:
						allow_reconnect = True

		if len(potential_rs) + 1 > len(extension_vertex.out_edges):
			if DEBUG: print '... compute_targets_by_best: potential_rs={}'.format([rs.id for rs in potential_rs])
			expected_positions = []
			for rs in potential_rs:
				pos = rs.closest_pos(extension_vertex.point)
				if path.is_explored(pos):
					continue
				rs_follow_positions = graph.follow_graph(pos, segment_length, explored_node_pairs=path.explored_pairs)
				if DEBUG: print '... compute_targets_by_best: rs {}: closest pos to extension point {} is on edge {}@{} at {}'.format(rs.id, extension_vertex.point, pos.edge.id, pos.distance, pos.point())
				for rs_follow_pos in rs_follow_positions:
					if DEBUG: print '... compute_targets_by_best: rs {}: ... {}@{} at {}'.format(rs.id, rs_follow_pos.edge.id, rs_follow_pos.distance, rs_follow_pos.point())
				expected_positions.extend(rs_follow_positions)
			set_by_positions(expected_positions)
		else:
			if DEBUG: print '... compute_targets_by_best: found {} potential rs but already have {} outgoing edges'.format(len(potential_rs), len(extension_vertex.out_edges))
	else:
		if DEBUG: print '... compute_targets_by_best: edge_pos is None'

	return angle_targets, allow_reconnect

def compute_targets_by_cgan(path, extension_vertex, segment_length, angle_outputs):
	rotation_target = 64

	def best_angle_to_pos(pos):
		angle_points = [get_next_point(extension_vertex.point, angle_bucket, segment_length) for angle_bucket in xrange(64)]
		distances = [angle_point.distance(pos.point()) for angle_point in angle_points]
		point_angle = numpy.argmin(distances) * math.pi * 2 / 64.0 - math.pi
		edge_angle = geom.Point(1, 0).signed_angle(pos.edge.segment().vector())
		avg_vector = vector_from_angle(point_angle).add(vector_from_angle(edge_angle))
		avg_angle = geom.Point(1, 0).signed_angle(avg_vector)
		return int((avg_angle + math.pi) * 64.0 / math.pi / 2)

	if extension_vertex.edge_pos is not None:
		cur_edge = extension_vertex.edge_pos.edge
		cur_rs = path.gc.edge_to_rs[cur_edge.id]
		prev_rs = None

		if len(extension_vertex.in_edges) >= 1:
			prev_vertex = extension_vertex.in_edges[0].src
			if prev_vertex.edge_pos is not None:
				prev_edge = prev_vertex.edge_pos.edge
				prev_rs = path.gc.edge_to_rs[prev_edge.id]

		potential_rs = []
		if cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance + segment_length < cur_rs.length():
			potential_rs.append(cur_rs)
		else:
			for rs in cur_rs.out_rs(path.gc.edge_to_rs):
				if rs == cur_rs or rs.is_opposite(cur_rs):
					continue
				potential_rs.append(rs)
		if cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance < segment_length / 2 and prev_rs is not None:
			for rs in cur_rs.in_rs(path.gc.edge_to_rs):
				if rs == cur_rs or rs.is_opposite(cur_rs) or rs == prev_rs or rs.is_opposite(prev_rs):
					continue

				# add the opposite of this rs so that we are going away from extension_vertex
				opposite_rs = path.gc.edge_to_rs[rs.edges[0].get_opposite_edge().id]
				potential_rs.append(opposite_rs)

		if len(potential_rs) + 1 > len(extension_vertex.out_edges):
			potential_rs = [rs for rs in potential_rs if not path.is_explored(rs.edges[0])]
			if DEBUG: print '... compute_targets_by_best: potential_rs={}'.format([rs.id for rs in potential_rs])
			expected_positions = []
			for rs in potential_rs:
				pos = rs.closest_pos(extension_vertex.point)
				rs_follow_positions = graph.follow_graph(pos, segment_length, explored_node_pairs=path.explored_pairs)
				if DEBUG: print '... compute_targets_by_best: rs {}: closest pos to extension point {} is on edge {}@{} at {}'.format(rs.id, extension_vertex.point, pos.edge.id, pos.distance, pos.point())
				for rs_follow_pos in rs_follow_positions:
					if DEBUG: print '... compute_targets_by_best: rs {}: ... {}@{} at {}'.format(rs.id, rs_follow_pos.edge.id, rs_follow_pos.distance, rs_follow_pos.point())
				expected_positions.extend(rs_follow_positions)

			angle_buckets = [best_angle_to_pos(pos) for pos in expected_positions]
			if angle_buckets:
				rotation_target = random.choice(angle_buckets)

	rotation_amount = rotation_target - numpy.argmax(angle_outputs)
	angle_targets = numpy.zeros((65,), dtype='float32')
	for i in xrange(65):
		angle_targets[i] = angle_outputs[(i - rotation_amount + 65) % 65]

	angle_targets_copy = numpy.zeros((65,), dtype='float32')
	angle_targets_copy[:] = angle_targets[:]
	angle_targets_copy[rotation_target] = 0
	if numpy.max(angle_targets) - numpy.max(angle_targets_copy) < 0.5:
		if numpy.max(angle_targets) >= 0.5:
			angle_targets = angle_targets * 0.5
		angle_targets[rotation_target] += random.random() * 0.1 + 0.4

	return angle_targets

def helper_compute_viz_points(path, extension_vertex, segment_length):
	if extension_vertex.edge_pos is not None:
		cur_edge = extension_vertex.edge_pos.edge
		cur_rs = path.gc.edge_to_rs[cur_edge.id]
		prev_rs = None

		if len(extension_vertex.in_edges) >= 1:
			prev_vertex = extension_vertex.in_edges[0].src
			if prev_vertex.edge_pos is not None:
				prev_edge = prev_vertex.edge_pos.edge
				prev_rs = path.gc.edge_to_rs[prev_edge.id]

		potential_rs = []
		if cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance + segment_length < cur_rs.length():
			potential_rs.append(cur_rs)
		else:
			for rs in cur_rs.out_rs(path.gc.edge_to_rs):
				if rs == cur_rs or rs.is_opposite(cur_rs):
					continue
				potential_rs.append(rs)
		if cur_rs.edge_distances[cur_edge.id] + extension_vertex.edge_pos.distance < segment_length / 2 and prev_rs is not None:
			for rs in cur_rs.in_rs(path.gc.edge_to_rs):
				if rs == cur_rs or rs.is_opposite(cur_rs) or rs == prev_rs or rs.is_opposite(prev_rs):
					continue

				# add the opposite of this rs so that we are going away from extension_vertex
				opposite_rs = path.gc.edge_to_rs[rs.edges[0].get_opposite_edge().id]
				potential_rs.append(opposite_rs)

		mm_point = extension_vertex.edge_pos.point()
		nx_points = []

		if len(potential_rs) + 1 > len(extension_vertex.out_edges):
			if DEBUG: print '... compute_targets_by_best: potential_rs={}'.format([rs.id for rs in potential_rs])
			expected_positions = []
			for rs in potential_rs:
				pos = rs.closest_pos(extension_vertex.point)
				if path.is_explored(pos):
					continue
				rs_follow_positions = graph.follow_graph(pos, segment_length, explored_node_pairs=path.explored_pairs)
				if DEBUG: print '... compute_targets_by_best: rs {}: closest pos to extension point {} is on edge {}@{} at {}'.format(rs.id, extension_vertex.point, pos.edge.id, pos.distance, pos.point())
				for rs_follow_pos in rs_follow_positions:
					if DEBUG: print '... compute_targets_by_best: rs {}: ... {}@{} at {}'.format(rs.id, rs_follow_pos.edge.id, rs_follow_pos.distance, rs_follow_pos.point())
				nx_points.extend([pos.point() for pos in rs_follow_positions])
		else:
			if DEBUG: print '... compute_targets_by_best: found {} potential rs but already have {} outgoing edges'.format(len(potential_rs), len(extension_vertex.out_edges))

		return {
			'mm': mm_point,
			'nx': nx_points,
		}
	else:
		if DEBUG: print '... compute_targets_by_best: edge_pos is None'

	return None
