from discoverlib import geom, graph
import model_m6a as model
import tileloader as tileloader

import numpy
import math
import os
import os.path
import random
import scipy.ndimage
import sys
import tensorflow as tf
import time

model_path = sys.argv[1]
old_tile_path = sys.argv[2]
new_tile_path = sys.argv[3]
graph_path = sys.argv[4]
angle_path = sys.argv[5]

MODEL_BASE = model_path
tileloader.REGIONS = ['mass']
tileloader.TRAINING_REGIONS = tileloader.REGIONS
tileloader.tile_dir = [
	old_tile_path,
	new_tile_path,
]
tileloader.graph_dir = graph_path
tileloader.angles_dir = angle_path

WINDOW_SIZE = 256
NUM_TRAIN_TILES = 1024
TILE_SIZE = 4096
RECT_OVERRIDE = None
NUM_BUCKETS = 64
MASK_NEAR_ROADS = False

tileloader.tile_size = 4096
tileloader.window_size = 256

tiles = tileloader.Tiles(2, 20, NUM_TRAIN_TILES+8, 'sat')
tiles.prepare_training()

train_tiles = list(tiles.train_tiles)
random.shuffle(train_tiles)
num_val = len(train_tiles)//10
val_tiles = train_tiles[0:num_val]
train_tiles = train_tiles[num_val:]

print('pick {} train tiles from {}'.format(len(train_tiles), len(tiles.train_tiles)))

# initialize model and session
print('initializing model')
m = model.Model(input_channels=3, bn=True)
session = tf.Session()
model_path = os.path.join(MODEL_BASE, 'model_latest/model')
best_path = os.path.join(MODEL_BASE, 'model_best/model')
if os.path.isfile(model_path + '.meta'):
	print('... loading existing model')
	m.saver.restore(session, model_path)
else:
	print('... initializing a new model')
	session.run(m.init_op)

def get_tile_rect(tile):
	if RECT_OVERRIDE:
		return RECT_OVERRIDE
	p = geom.Point(tile.x, tile.y)
	return geom.Rectangle(
		p.scale(TILE_SIZE),
		p.add(geom.Point(1, 1)).scale(TILE_SIZE)
	)

def get_tile_example(tile, tries=10):
	rect = get_tile_rect(tile)

	# pick origin: must be multiple of the output scale
	origin = geom.Point(random.randint(0, rect.lengths().x//4 - WINDOW_SIZE//4), random.randint(0, rect.lengths().y//4 - WINDOW_SIZE//4))
	origin = origin.scale(4)
	origin = origin.add(rect.start)

	tile_origin = origin.sub(rect.start)
	big_ims = tiles.cache.get_window(tile.region, rect, geom.Rectangle(tile_origin, tile_origin.add(geom.Point(WINDOW_SIZE, WINDOW_SIZE))))
	if len(tileloader.get_tile_keys()) > 1:
		inputs = [big_ims[key] for key in tileloader.get_tile_keys()]
		#input = numpy.concatenate(inputs, axis=2).astype('float32') / 255.0
		input = random.choice(inputs).astype('float32') / 255.0
	else:
		input = big_ims['input'].astype('float32') / 255.0
	target = big_ims['angles'].astype('float32') / 255.0
	if numpy.count_nonzero(target.max(axis=2)) < 64 and tries > 0:
		#return get_tile_example(tile, tries - 1)
		return None
	example = {
		'region': tile.region,
		'origin': origin,
		'input': input,
		'target': target,
	}
	if MASK_NEAR_ROADS:
		mask = target.max(axis=2) > 0
		mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=9)
		example['mask'] = mask
	return example

def get_example(traintest='train'):
	while True:
		if traintest == 'train':
			tile = random.choice(train_tiles)
		elif traintest == 'test':
			tile = random.choice(val_tiles)

		example = get_tile_example(tile)
		if example is not None:
			return example

val_examples = [get_example('test') for _ in range(2048)]

def vis_example(example, outputs=None):
	x = numpy.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype='uint8')
	x[:, :, :] = example['input'] * 255
	x[WINDOW_SIZE//2-2:WINDOW_SIZE//2+2, WINDOW_SIZE//2-2:WINDOW_SIZE//2+2, :] = 255

	gc = tiles.get_gc(example['region'])
	rect = geom.Rectangle(example['origin'], example['origin'].add(geom.Point(WINDOW_SIZE, WINDOW_SIZE)))
	for edge in gc.edge_index.search(rect):
		start = edge.src.point
		end = edge.dst.point
		for p in geom.draw_line(start.sub(example['origin']), end.sub(example['origin']), geom.Point(WINDOW_SIZE, WINDOW_SIZE)):
			x[p.x, p.y, 0:2] = 0
			x[p.x, p.y, 2] = 255

	for i in range(WINDOW_SIZE):
		for j in range(WINDOW_SIZE):
			di = i - WINDOW_SIZE//2
			dj = j - WINDOW_SIZE//2
			d = math.sqrt(di * di + dj * dj)
			a = int((math.atan2(dj, di) - math.atan2(0, 1) + math.pi) * NUM_BUCKETS / 2 / math.pi)
			if a >= NUM_BUCKETS:
				a = NUM_BUCKETS - 1
			elif a < 0:
				a = 0
			elif d > 100 and d <= 120 and example['target'] is not None:
				x[i, j, 0] = example['target'][WINDOW_SIZE//8, WINDOW_SIZE//8, a] * 255
				x[i, j, 1] = example['target'][WINDOW_SIZE//8, WINDOW_SIZE//8, a] * 255
				x[i, j, 2] = 0
			elif d > 70 and d <= 90 and outputs is not None:
				x[i, j, 0] = outputs[WINDOW_SIZE//8, WINDOW_SIZE//8, a] * 255
				x[i, j, 1] = outputs[WINDOW_SIZE//8, WINDOW_SIZE//8, a] * 255
				x[i, j, 2] = 0
	return x

def get_learning_rate(epoch):
	if epoch < 100:
		return 1e-4
	else:
		return 1e-5

best_loss = None

for epoch in range(200):
	start_time = time.time()
	train_losses = []
	for _ in range(1024):
		examples = [get_example('train') for _ in range(model.BATCH_SIZE)]
		feed_dict = {
			m.is_training: True,
			m.inputs: [example['input'] for example in examples],
			m.targets: [example['target'] for example in examples],
			m.learning_rate: get_learning_rate(epoch),
		}
		if MASK_NEAR_ROADS:
			feed_dict[m.mask] = [example['mask'] for example in examples]
		_, loss = session.run([m.optimizer, m.loss], feed_dict=feed_dict)
		train_losses.append(loss)

	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_examples), model.BATCH_SIZE):
		examples = val_examples[i:i+model.BATCH_SIZE]
		feed_dict = {
			m.is_training: False,
			m.inputs: [example['input'] for example in examples],
			m.targets: [example['target'] for example in examples],
		}
		if MASK_NEAR_ROADS:
			feed_dict[m.mask] = [example['mask'] for example in examples]
		loss = session.run([m.loss], feed_dict=feed_dict)
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

	m.saver.save(session, model_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)
