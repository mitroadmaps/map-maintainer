from discoverlib import geom
from discoverlib import graph
import model_utils

import json
import numpy
import os
import random
import rtree
import skimage.io
import time

# This is now set in run_m6.py or infer_change.py.
tile_dir = None
graph_dir = None
angles_dir = None
TRAINING_REGIONS = []
REGIONS = []

tile_size = 4096
window_size = 256

def get_tile_dirs():
	if isinstance(tile_dir, str):
		return [tile_dir]
	else:
		return tile_dir

def get_tile_keys():
	keys = []
	for i in range(len(get_tile_dirs())):
		if i == 0:
			keys.append('input')
		else:
			keys.append('input{}'.format(i))
	return keys

def load_tile(region, i, j, mode='all'):
	d = {}
	for pathIdx, path in enumerate(get_tile_dirs()):
		prefix = os.path.join(path, '{}_{}_{}_'.format(region, i, j))
		sat_im = skimage.io.imread(prefix + 'sat.jpg')
		if sat_im.shape == (tile_size, tile_size, 4):
			sat_im = sat_im[:, :, 0:3]
		sat_im = sat_im.swapaxes(0, 1)
		if pathIdx == 0:
			d['input'] = sat_im
		else:
			d['input{}'.format(pathIdx)] = sat_im
	if angles_dir:
		angle_path = os.path.join(angles_dir, '{}_{}_{}.bin'.format(region, i, j))
		angle_im = numpy.fromfile(angle_path, dtype='uint8')
		angle_im = angle_im.reshape(tile_size//4, tile_size//4, 64)
		d['angles'] = angle_im

	return d

def load_rect(region, rect, load_func=load_tile, mode='all'):
	# special case for fast load: rect is single tile
	if rect.start.x % tile_size == 0 and rect.start.y % tile_size == 0 and rect.end.x % tile_size == 0 and rect.end.y % tile_size == 0 and rect.end.x - rect.start.x == tile_size and rect.end.y - rect.start.y == tile_size:
		return load_func(region, rect.start.x // tile_size, rect.start.y // tile_size, mode=mode)

	tile_rect = geom.Rectangle(
		geom.Point(rect.start.x // tile_size, rect.start.y // tile_size),
		geom.Point((rect.end.x - 1) // tile_size + 1, (rect.end.y - 1) // tile_size + 1)
	)
	full_rect = geom.Rectangle(
		tile_rect.start.scale(tile_size),
		tile_rect.end.scale(tile_size)
	)
	full_ims = {}

	for i in range(tile_rect.start.x, tile_rect.end.x):
		for j in range(tile_rect.start.y, tile_rect.end.y):
			p = geom.Point(i - tile_rect.start.x, j - tile_rect.start.y).scale(tile_size)
			tile_ims = load_func(region, i, j, mode=mode)
			for k, im in tile_ims.iteritems():
				scale = tile_size // im.shape[0]
				if k not in full_ims:
					full_ims[k] = numpy.zeros((full_rect.lengths().x // scale, full_rect.lengths().y // scale, im.shape[2]), dtype='uint8')
				full_ims[k][p.x//scale:(p.x+tile_size)//scale, p.y//scale:(p.y+tile_size)//scale, :] = im

	crop_rect = geom.Rectangle(
		rect.start.sub(full_rect.start),
		rect.end.sub(full_rect.start)
	)
	for k in full_ims:
		scale = (full_rect.end.x - full_rect.start.x) // full_ims[k].shape[0]
		full_ims[k] = full_ims[k][crop_rect.start.x//scale:crop_rect.end.x//scale, crop_rect.start.y//scale:crop_rect.end.y//scale, :]
	return full_ims

class TileCache(object):
	def __init__(self, limit=128, mode='all'):
		self.limit = limit
		self.mode = mode
		self.cache = {}
		self.last_used = {}

	def reduce_to(self, limit):
		while len(self.cache) > limit:
			best_k = None
			best_used = None
			for k in self.cache:
				if best_k is None or self.last_used.get(k, 0) < best_used:
					best_k = k
					best_used = self.last_used.get(k, 0)
			del self.cache[best_k]

	def get(self, region, rect):
		k = '{}.{}.{}.{}.{}'.format(region, rect.start.x, rect.start.y, rect.end.x, rect.end.y)
		if k not in self.cache:
			self.reduce_to(self.limit - 1)
			self.cache[k] = load_rect(region, rect, mode=self.mode)
		self.last_used[k] = time.time()
		return self.cache[k]

	def get_window(self, region, big_rect, small_rect):
		big_dict = self.get(region, big_rect)
		small_dict = {}
		for k, v in big_dict.items():
			scale = tile_size // v.shape[0]
			small_dict[k] = v[small_rect.start.x//scale:small_rect.end.x//scale, small_rect.start.y//scale:small_rect.end.y//scale, :]
		return small_dict

def get_tile_list():
	tiles = []
	for fname in os.listdir(get_tile_dirs()[0]):
		if not fname.endswith('_sat.jpg'):
			continue
		parts = fname.split('_sat.jpg')[0].split('_')
		region = parts[0]
		x, y = int(parts[1]), int(parts[2])
		tile = geom.Point(x, y)
		tile.region = region
		tiles.append(tile)
	if angles_dir:
		# filter tiles for only those that are in angles_dir
		angle_keys = set(os.listdir(angles_dir))
		tiles = [tile for tile in tiles if '{}_{}_{}.bin'.format(tile.region, tile.x, tile.y) in angle_keys]
	return tiles

def get_input_channels_from_mode(tile_mode):
	if tile_mode == 'all':
		return 7
	elif tile_mode == 'sat':
		return 5
	elif tile_mode == 'gpsa':
		return 4

class Tiles(object):
	def __init__(self, paths_per_tile_axis, segment_length, parallel_tiles, tile_mode):
		self.search_rect_size = tile_size // paths_per_tile_axis
		self.segment_length = segment_length
		self.parallel_tiles = parallel_tiles
		self.tile_mode = tile_mode

		# load tile list
		# this is a list of point dicts (a point dict has keys 'x', 'y')
		# don't include test tiles
		print('reading tiles')
		self.all_tiles = get_tile_list()
		self.cache = TileCache(limit=self.parallel_tiles, mode=self.tile_mode)

	def prepare_training(self):
		def tile_filter(tile):
			if tile.region not in REGIONS:
				return False
			return True
		self.train_tiles = list(filter(tile_filter, self.all_tiles))

		old_len = len(self.train_tiles)
		self.train_tiles = [tile for tile in self.train_tiles if tile.region in TRAINING_REGIONS]
		print('go from {} to {} tiles after excluding regions'.format(old_len, len(self.train_tiles)))
		random.shuffle(self.train_tiles)

	def get_tile_data(self, region, rect):
		midpoint = rect.start.add(rect.end.sub(rect.start).scale(0.5))
		x = midpoint.x // tile_size
		y = midpoint.y // tile_size
		k = '{}_{}_{}'.format(region, x, y)
		return {
			'region': region,
			'rect': rect,
			'search_rect': rect.add_tol(-window_size//2),
			'cache': self.cache,
			'starting_locations': [],
		}

	def num_tiles(self):
		return len(self.train_tiles)

	def num_input_channels(self):
		return get_input_channels_from_mode(self.tile_mode)

	def get_training_tile_data(self, tile_idx):
		tile = self.train_tiles[tile_idx]
		rect = geom.Rectangle(
			tile.scale(tile_size),
			tile.add(geom.Point(1, 1)).scale(tile_size)
		)

		if tries < 3:
			search_rect_x = random.randint(window_size//2, tile_size - window_size//2 - self.search_rect_size)
			search_rect_y = random.randint(window_size//2, tile_size - window_size//2 - self.search_rect_size)
			search_rect = geom.Rectangle(
				rect.start.add(geom.Point(search_rect_x, search_rect_y)),
				rect.start.add(geom.Point(search_rect_x, search_rect_y)).add(geom.Point(self.search_rect_size, self.search_rect_size)),
			)

		return {
			'region': tile.region,
			'rect': rect,
			'search_rect': search_rect,
			'cache': self.cache,
			'starting_locations': [],
		}
