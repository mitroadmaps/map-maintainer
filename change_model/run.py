import model as model

import numpy
import os
from PIL import Image
import random
import skimage.io, skimage.morphology
import subprocess
import sys
import tensorflow as tf
import time

sys.path.append('../changeseeking_tracing/')
from discoverlib import geom, graph

model_path = sys.argv[1]
old_tile_path = sys.argv[2]
new_tile_path = sys.argv[3]
graph_path = sys.argv[4]

SIZE = 512
PATH = model_path
SAT_PATHS = [old_tile_path, new_tile_path]
OSM_PATH = graph_path
REGIONS = ['mass']

print('computing tiles')
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
keys = random.sample(keys, 400)

def load_tile(k):
	print('... load {}'.format(k))
	region, x, y = k.split('_')

	ims = []
	for sat_path in SAT_PATHS:
		sat_fname = '{}/{}_sat.jpg'.format(sat_path, k)
		im = skimage.io.imread(sat_fname)[:, :, 0:3]
		ims.append(im)

	g = graph.read_graph('{}/{}.graph'.format(OSM_PATH, k))
	road_segments, edge_to_rs = graph.get_graph_road_segments(g)

	return (ims, g, road_segments, edge_to_rs, k)

print('reading {} tiles'.format(len(keys)))
tiles = [load_tile(k) for k in keys]
tiles = [tile for tile in tiles if tile is not None]
random.shuffle(tiles)
num_val_tiles = len(tiles)//10 + 1
val_tiles = tiles[0:num_val_tiles]
train_tiles = tiles[num_val_tiles:]
print('... done!')

def bfs(rs0, sizethresh, edge_to_rs):
	explored = set()
	q = [rs0]
	rect = None
	rs_list = []
	while len(q) > 0:
		rs = q[0]
		q = q[1:]

		rs_list.append(rs)
		if rect is None:
			rect = rs.bounds()
		else:
			rect = rect.extend_rect(rs.bounds())
		if rect.lengths().x > sizethresh or rect.lengths().y > sizethresh:
			break

		for other_rs in rs.in_rs(edge_to_rs) + rs.out_rs(edge_to_rs):
			if other_rs.id in explored:
				continue
			explored.add(other_rs.id)
			q.append(other_rs)

	return rs_list, rect

def extract(tiles):
	def sample_disjoint(origin):
		while True:
			i = random.randint(0, 4096-SIZE)
			j = random.randint(0, 4096-SIZE)
			if i >= origin.x and i < origin.x+SIZE and j >= origin.y and j < origin.y+SIZE:
				continue
			return geom.Point(i, j)

	while True:
		ims, g, road_segments, edge_to_rs, _ = random.choice(tiles)
		im_rect = geom.Rectangle(geom.Point(0, 0), geom.Point(4096, 4096))
		ok_origin_rect = geom.Rectangle(geom.Point(0, 0), geom.Point(4096-SIZE, 4096-SIZE))

		# (1) find an osm mask (i1, j1)
		# (2) decide whether to:
		#     * 45% compare (i1, j1) to itself
		#     * 45% compare (i1, j1) to some (i2, j2)
		#     * 10% compare some (i2, j2) to itself using the first mask
		rs0 = random.choice(road_segments)
		if not im_rect.contains(rs0.src().point):
			continue

		sizethresh = random.randint(64, 256)
		rs_list, bfs_rect = bfs(rs0, sizethresh, edge_to_rs)
		origin = ok_origin_rect.clip(rs0.src().point.sub(geom.Point(SIZE//2, SIZE//2)))
		mask = numpy.zeros((SIZE, SIZE), dtype='bool')
		for rs in rs_list:
			for edge in rs.edges:
				src = edge.src.point.sub(origin)
				dst = edge.dst.point.sub(origin)
				for p in geom.draw_line(src, dst, geom.Point(SIZE, SIZE)):
					mask[p.y, p.x] = True
		mask = skimage.morphology.binary_dilation(mask, selem=skimage.morphology.disk(15))
		mask = mask.astype('uint8').reshape(SIZE, SIZE, 1)

		rand = random.random()
		if rand < 0.45:
			im1 = ims[0][origin.y:origin.y+SIZE, origin.x:origin.x+SIZE]
			im2 = ims[1][origin.y:origin.y+SIZE, origin.x:origin.x+SIZE]
			label = 0
		elif rand < 0.9:
			other_point = sample_disjoint(origin)
			if random.random() < 0.5:
				im1 = ims[0][origin.y:origin.y+SIZE, origin.x:origin.x+SIZE]
				im2 = ims[1][other_point.y:other_point.y+SIZE, other_point.x:other_point.x+SIZE]
			else:
				im1 = ims[0][other_point.y:other_point.y+SIZE, other_point.x:other_point.x+SIZE]
				im2 = ims[1][origin.y:origin.y+SIZE, origin.x:origin.x+SIZE]
			label = 1
		else:
			other_point = sample_disjoint(origin)
			im1 = ims[0][other_point.y:other_point.y+SIZE, other_point.x:other_point.x+SIZE]
			im2 = ims[1][other_point.y:other_point.y+SIZE, other_point.x:other_point.x+SIZE]
			label = 0

		mask_tile = numpy.concatenate([mask, mask, mask], axis=2)
		cat_im = numpy.concatenate([im1*mask_tile, im2*mask_tile, mask], axis=2)
		return cat_im, label

val_rects = [extract(val_tiles) for _ in range(1024)]

print('initializing model')
m = model.Model(size=SIZE)
session = tf.Session()
session.run(m.init_op)
latest_path = '{}/model_latest/model'.format(PATH)
best_path = '{}/model_best/model'.format(PATH)

print('begin training')
best_loss = None

def vis(rects):
	for i, (im, label) in enumerate(rects):
		skimage.io.imsave('/home/ubuntu/vis/{}-{}-a.png'.format(i, label), im[:, :, 0:3])
		skimage.io.imsave('/home/ubuntu/vis/{}-{}-b.png'.format(i, label), im[:, :, 3:6])

def get_learning_rate(epoch):
	if epoch < 100:
		return 1e-4
	else:
		return 1e-5

for epoch in range(200):
	start_time = time.time()
	train_losses = []
	for _ in range(128):
		batch_rects = [extract(train_tiles) for _ in range(model.BATCH_SIZE)]
		_, loss = session.run([m.optimizer, m.loss], feed_dict={
			m.is_training: True,
			m.inputs: [tile[0] for tile in batch_rects],
			m.targets: [tile[1] for tile in batch_rects],
			m.learning_rate: get_learning_rate(epoch),
		})
		train_losses.append(loss)
	train_loss = numpy.mean(train_losses)
	train_time = time.time()

	val_losses = []
	for i in range(0, len(val_rects), model.BATCH_SIZE):
		batch_rects = val_rects[i:i+model.BATCH_SIZE]
		outputs, loss = session.run([m.outputs, m.loss], feed_dict={
			m.is_training: False,
			m.inputs: [tile[0] for tile in batch_rects],
			m.targets: [tile[1] for tile in batch_rects],
		})
		val_losses.append(loss)

	val_loss = numpy.mean(val_losses)
	val_time = time.time()

	print('iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}'.format(epoch, int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss))

	m.saver.save(session, latest_path)
	if best_loss is None or val_loss < best_loss:
		best_loss = val_loss
		m.saver.save(session, best_path)

def test(rects):
	for i in range(0, len(rects), model.BATCH_SIZE):
		batch_rects = val_rects[i:i+model.BATCH_SIZE]
		outputs = session.run(m.outputs, feed_dict={
			m.is_training: False,
			m.inputs: [tile[0] for tile in batch_rects],
		})
		for j, (im, label) in enumerate(batch_rects):
			output = outputs[j]
			skimage.io.imsave('/home/ubuntu/vis/{}-{}-{}-a.png'.format(i+j, label, output), im[:, :, 0:3])
			skimage.io.imsave('/home/ubuntu/vis/{}-{}-{}-b.png'.format(i+j, label, output), im[:, :, 3:6])
