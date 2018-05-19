import h5py 
import numpy
import tensorflow as tf
import time
import sys
import math

BATCH_SIZE = 50
MAX_EPOCH = 100
VAL_STEP = 10
GPU_INDEX = 0
BASE_LEARNING_RATE = 1e-2
POOL_SIZE = 3
NUM_CONV_LAYERS = 5
NUM_FC_LAYERS = 0
NUM_FEATURE_CHANNELS = 256
NUM_FC_CHANNELS = 64
RESAMPLE = 10
NUM_POINT = 243
NUM_FEATURE_CHANNELS = 256

TRAIN_PATH = [
	'bimnet10_train.h5',
]
VALIDATION_PATH = [
	'bimnet10_test.h5',
#	'bimnet10_test_deform.h5',
]
TEST = False
LOAD_PATH = 'models/bimnet10/model.ckpt'
MODEL_PATH = 'models/model.ckpt'

def batch_norm_template(i,inputs, is_training, moments_dims):
	with tf.variable_scope('conv'+str(i)+'/bn') as sc:
		num_channels = inputs.get_shape()[-1].value
		beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
			name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
			name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.9)
		ema_apply_op = tf.cond(is_training,
			lambda: ema.apply([batch_mean, batch_var]),
			lambda: tf.no_op())

		def mean_var_with_update():
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(is_training,
			mean_var_with_update,
			lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
	return normed

class BimNet():
	def __init__(self,batch_size,num_point,num_class):
		#inputs
		self.pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))	
		self.is_training_pl = tf.placeholder(tf.bool, shape=())
		self.input = tf.expand_dims(self.pointclouds_pl,-1)
		self.conv = [None] * NUM_CONV_LAYERS
		self.kernel = [None] * NUM_CONV_LAYERS
		self.bias = [None] * NUM_CONV_LAYERS
		self.pool = [None] * NUM_CONV_LAYERS
		self.fc = [None] * (NUM_FC_LAYERS + 1)
		self.fc_weights = [None] * (NUM_FC_LAYERS + 1)
		self.fc_bias = [None] * (NUM_FC_LAYERS + 1)

		#hidden layers
		for i in range(NUM_CONV_LAYERS):
			conv_param = NUM_FEATURE_CHANNELS
			self.kernel[i] = tf.get_variable('conv'+str(i)+'/weights', [1,3 if i==0 else 1, 1 if i==0 else NUM_FEATURE_CHANNELS, conv_param], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.bias[i] = tf.get_variable('conv'+str(i)+'/biases', [conv_param], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
			self.conv[i] = tf.nn.conv2d(self.input if i==0 else self.pool[i-1], self.kernel[i], [1, 1, 1, 1], padding='VALID')
			self.conv[i] = tf.nn.bias_add(self.conv[i], self.bias[i])
			self.conv[i] = batch_norm_template(i,self.conv[i],self.is_training_pl,[0,1,2])
			self.conv[i] = tf.nn.relu(self.conv[i])

			pool_param = num_point/(POOL_SIZE**(NUM_CONV_LAYERS-1)) if i==NUM_CONV_LAYERS-1 else POOL_SIZE
			self.pool[i] = tf.nn.avg_pool(self.conv[i],ksize=[1, pool_param, 1, 1],strides=[1, pool_param, 1, 1], padding='VALID', name='pool'+str(i))

		self.fc[0] = tf.reshape(self.pool[-1], [batch_size, -1])
		for i in range(NUM_FC_LAYERS):
			self.fc_weights[i] = tf.get_variable('fc_weights'+str(i), [self.fc[i].get_shape().as_list()[1], NUM_FC_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc_bias[i] = tf.get_variable('fc_bias'+str(i), [NUM_FC_CHANNELS], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.fc[i+1] = tf.matmul(self.fc[i], self.fc_weights[i])
			self.fc[i+1] = tf.nn.bias_add(self.fc[i+1], self.fc_bias[i])
			self.fc[i+1] = batch_norm_template(self.fc[i+1],self.is_training_pl,[0,])
			self.fc[i+1] = tf.nn.relu(self.fc[i+1])

		#output
		self.fc_weights[-1] = tf.get_variable('pred/weights'+str(NUM_FC_LAYERS), [self.fc[-1].get_shape().as_list()[1], num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.fc_bias[-1] = tf.get_variable('pred/biases'+str(NUM_FC_LAYERS), [num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.output = tf.matmul(self.fc[-1], self.fc_weights[-1])
		self.output = tf.nn.bias_add(self.output, self.fc_bias[-1])
		
		#loss functions
		#weight decay loss
		self.weight_loss = 0
		for i in range(NUM_CONV_LAYERS):
			self.weight_loss += tf.nn.l2_loss(self.kernel[i])
			self.weight_loss += tf.nn.l2_loss(self.bias[i])
		for i in range(NUM_FC_LAYERS + 1):
			self.weight_loss += tf.nn.l2_loss(self.fc_weights[i])
			self.weight_loss += tf.nn.l2_loss(self.fc_bias[i])

		self.weight_loss *= 0.000
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels_pl))
		self.loss = self.class_loss + self.weight_loss
		self.correct = tf.equal(tf.argmax(self.output, 1), tf.to_int64(self.labels_pl))
		self.accuracy = tf.reduce_sum(tf.cast(self.correct, tf.float32)) / float(batch_size)

		#optimizer
		self.batch = tf.Variable(0)
		self.learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE,self.batch,1000,0.5,staircase=True)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)

def readData(filename):
	f = h5py.File(filename)
	points = f['data'][:]
	for i in range(len(points)):
		points[i,:,:] =  numpy.array(sorted(points[i,:,:],key=lambda x:x[2]))
	labels = f['label'][:]
	return points, labels

def pca_norm(batch_pcd):
	for pcd in batch_pcd:
		centroid = pcd.mean(axis=0)
		pcd -= centroid
		R = numpy.sum(pcd**2,axis=1)
		id_max = numpy.argmax(R)
		principal_vector = pcd[id_max,:2].copy()
		principal_vector /= numpy.linalg.norm(principal_vector)
		principal_vector_perp = numpy.array([-principal_vector[1],principal_vector[0]])
		R = numpy.vstack((principal_vector,principal_vector_perp)).transpose()
		pcd[:,:2] = pcd[:,:2].dot(R)

points = None
labels = None
val_points = None
val_labels = None
for path in TRAIN_PATH:
	if points is None:
		points, labels = readData(path)
	else:
		p, l = readData(path)
		points = numpy.vstack((points,p))
		labels = numpy.hstack((labels,l))
for path in VALIDATION_PATH:
	if val_points is None:
		val_points, val_labels = readData(path)
	else:
		p, l = readData(path)
		val_points = numpy.vstack((val_points,p))
		val_labels = numpy.hstack((val_labels,l))

new_val_points = []
for i in range(len(val_points)):
	for j in range(RESAMPLE):
		subset = sorted(numpy.random.choice(val_points.shape[1], NUM_POINT, replace=False))
		new_val_points.append(val_points[i, subset, :3])
repeat_labels = numpy.repeat(val_labels, RESAMPLE)
val_points = numpy.array(new_val_points)

NUM_CLASSES = max(labels) + 1
print('Points',points.shape)
print('Labels',labels.shape,'Classes',NUM_CLASSES)
print('Validation Points',val_points.shape)
print('Validation Labels',val_labels.shape)

with tf.Graph().as_default():
	with tf.device('/gpu:'+str(GPU_INDEX)):
		net = BimNet(BATCH_SIZE,NUM_POINT,NUM_CLASSES) 
		saver = tf.train.Saver()

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)
		if TEST:
			saver.restore(sess, MODEL_PATH)
			MAX_EPOCH = 1
		else:
			init = tf.global_variables_initializer()
			sess.run(init, {net.is_training_pl: True})

		input_points = numpy.zeros((BATCH_SIZE, NUM_POINT, 3))
		for epoch in range(MAX_EPOCH):
			if not TEST:
				#shuffle data
				idx = numpy.arange(len(labels))
				numpy.random.shuffle(idx)
				shuffled_points = points[idx, :, :3]
				shuffled_labels = labels[idx]

				#split into batches
				num_batches = int(len(labels) / BATCH_SIZE)
				losses = []
				accuracies = []
				for batch_id in range(num_batches):
					start_idx = batch_id * BATCH_SIZE
					end_idx = (batch_id + 1) * BATCH_SIZE
					if NUM_POINT == shuffled_points.shape[1]:
						input_points[:] = shuffled_points[start_idx:end_idx,:,:]
					else:
						for i in range(BATCH_SIZE):
							subset = sorted(numpy.random.choice(shuffled_points.shape[1], NUM_POINT, replace=False))
							input_points[i,:,:] = shuffled_points[start_idx+i, subset, :]
					pca_norm(input_points)
					feed_dict = {net.pointclouds_pl: input_points,
						net.labels_pl: shuffled_labels[start_idx:end_idx],
						net.is_training_pl: True}
					loss,accuracy,_ = sess.run([net.loss,net.accuracy,net.train_op],feed_dict=feed_dict)
					losses.append(loss)
					accuracies.append(accuracy)
				print('Epoch: %d Loss: %.3f Accuracy: %.3f'%(epoch, numpy.mean(losses),numpy.mean(accuracies)))

			if TEST or epoch % VAL_STEP == VAL_STEP - 1:
				#get validation loss
				num_batches = int(math.ceil(1.0 * len(repeat_labels) / BATCH_SIZE))
				predictions = []
				for batch_id in range(num_batches):
					start_idx = batch_id * BATCH_SIZE
					end_idx = (batch_id + 1) * BATCH_SIZE
					valid_idx = min(len(val_points),end_idx)
					input_points[:valid_idx-start_idx,:,:] = val_points[start_idx:valid_idx,:,:]
					pca_norm(input_points)
					feed_dict = {net.pointclouds_pl: input_points,
						net.is_training_pl: False}
					output, = sess.run([net.output], feed_dict=feed_dict)
					predictions.extend(output[:valid_idx-start_idx,:])

				val_correct = repeat_labels == numpy.argmax(predictions, axis=1)
				val_acc = 1.0 * numpy.sum(val_correct) / len(val_correct)
				resampled_predictions = numpy.reshape(predictions, (len(val_labels),RESAMPLE,NUM_CLASSES)).sum(axis=1)
				res_correct = val_labels == numpy.argmax(resampled_predictions, axis=1)
				res_acc = 1.0 * numpy.sum(res_correct) / len(res_correct)
				print('Validation: %d Accuracy: %.3f %.3f (with resampling)'%(epoch,val_acc, res_acc))

		saver.save(sess, MODEL_PATH)
