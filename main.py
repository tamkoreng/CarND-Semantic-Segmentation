import os.path
import helper
import warnings
import time

from distutils.version import LooseVersion
import tensorflow as tf
import numpy as np

import project_tests as tests

# architecture
IMAGE_SHAPE = (160, 576)
N_CLASSES = 2

# hyperparams
EPOCHS = 100
KEEP_PROB = 0.75
LR = 0.0001
BATCH_SIZE = 8
N_INFERENCE = 10

# paths
DATA_DIR = './data'
RUNS_DIR = './runs'
LOG_DIR = './train'


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # From L9.5
    def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
        return tf.random_normal(shape, dtype=dtype, seed=seed)

    def conv_1x1(x, num_outputs):
        kernel_size = 1
        stride = 1
        kernel_init = tf.truncated_normal_initializer(stddev=0.01)
        kernel_reg = tf.contrib.layers.l2_regularizer(0.001)
        return tf.layers.conv2d(x, num_outputs, kernel_size, stride, padding='SAME',
                                kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)
    
    layer3_encode = conv_1x1(vgg_layer3_out, num_classes)
    layer4_encode = conv_1x1(vgg_layer4_out, num_classes)
    layer7_encode = conv_1x1(vgg_layer7_out, num_classes)
    
    kernel_init = tf.truncated_normal_initializer(stddev=0.01)
    kernel_reg = tf.contrib.layers.l2_regularizer(0.001)

    decode1 = tf.layers.conv2d_transpose(layer7_encode, num_classes, 4, strides=(2, 2), padding='SAME',
                                         kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)
    
    # Skip connection to 4th pooling layer        
    decode2 = tf.add(decode1, layer4_encode)
    decode3 = tf.layers.conv2d_transpose(decode2, num_classes, 4, strides=(2, 2), padding='SAME', 
                                         kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)
    
    # Skip connection to 3rd pooling layer
    decode4 = tf.add(decode3, layer3_encode)
    decode5 = tf.layers.conv2d_transpose(decode4, num_classes, 16, strides=(8, 8), padding='SAME', 
                                         kernel_initializer=kernel_init, kernel_regularizer=kernel_reg)
    
    return decode5
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, 
             train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, 
             logits=None, # default value needed to pass test_train_nn
             summary_op=None, 
             saver=None, 
             run_dir=None,
             train_writer=None, 
             val_batches_fn=None, 
             val_writer=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)

    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate

    :param logits: Logits tensor.
    :param summary_op: TF summary operation.
    :param saver: TF saver object.
    :param run_dir: Path to run directory.
    :param train_writer: TF summary writer object for training data.
    :param val_batches_fn: Function to get batches of validation data.  Call using val_batches_fn(batch_size)
    :param val_writer: TF summary writer object for validation data.
    """
    log_epoch_step = 1
    val_epoch_step = 2
    
    if logits is None:
        # Original training function
        print("Training...")
        print()
        for epoch in range(epochs):
            print("EPOCH {} ...".format(epoch + 1))
            n_train = 0
            total_loss = 0
            for X_batch, Y_batch in get_batches_fn(batch_size):
                n_train += len(X_batch)
                feed_dict = {
                    input_image: X_batch,
                    correct_label: Y_batch,
                    keep_prob: KEEP_PROB,
                    learning_rate: LR,
                }
                loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict=feed_dict)
                total_loss += loss

            avg_loss = total_loss / n_train
            print("Avg loss = {:.5f}".format(avg_loss))

    else:
        # Training with validation
    
        # Create IOU metric (with help from http://ronny.rest/blog/post_2017_09_11_tf_metrics/)
        prediction_op = tf.argmax(tf.nn.softmax(logits), dimension=1) # logits flattened in first 3 dims
        tf_label = tf.argmax(tf.reshape(correct_label, (-1, N_CLASSES)), dimension=1)
    
        graph = tf.get_default_graph()        
        with graph.as_default():
            # Placeholders to take in batches of data
            tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])

            # Define the metric and update operations
            tf_metric, tf_metric_update = tf.metrics.mean_iou(tf_label,
                                                              tf_prediction,
                                                              N_CLASSES,
                                                              name="my_iou_metric")
            val_summary_op = tf.summary.scalar('IOU', tf_metric)

            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_iou_metric")

            # Define initializer to initialize/reset running variables
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    
        sess.run(running_vars_initializer)
            
        print("Training...")
        print()
        i_train_summary = 0
        i_val_summary = 0
        for epoch in range(epochs):
            print("EPOCH {} ...".format(epoch + 1))
            n_train = 0
            total_loss = 0
            for X_batch, Y_batch in get_batches_fn(batch_size):
                n_train += len(X_batch)
                feed_dict = {
                    input_image: X_batch,
                    correct_label: Y_batch,
                    keep_prob: KEEP_PROB,
                    learning_rate: LR,
                }
            
                i_train_summary += n_train
                loss, _, summary = sess.run([cross_entropy_loss, train_op, summary_op], feed_dict=feed_dict)
                total_loss += loss

            train_writer.add_summary(summary, i_train_summary)

            avg_loss = total_loss / n_train
            print("Avg loss = {:.5f}".format(avg_loss))
        
            # validation
            if (epoch + 1) % val_epoch_step == 0:
                print("Validating...")
                for X_batch, Y_batch in val_batches_fn(batch_size):
                    feed_dict = {
                        input_image: X_batch,
                        correct_label: Y_batch,
                        keep_prob: 1.0
                    }
                    pred = sess.run([prediction_op], feed_dict=feed_dict)
                    feed_dict = {
                        correct_label: Y_batch,
                        tf_prediction: np.reshape(pred, (-1,)),
                        keep_prob: 1.0
                    }
                    sess.run(tf_metric_update, feed_dict=feed_dict)
                
                iou = sess.run(tf_metric)
                val_summary = sess.run(val_summary_op)
                val_writer.add_summary(val_summary, i_train_summary)
                print("IOU = {:.5f}".format(iou))

            saver.save(sess, os.path.join(run_dir, 'model_{}'.format(epoch + 1)))        

        train_writer.close()
        val_writer.close()

tests.test_train_nn(train_nn)


#def run():
#    tests.test_for_kitti_dataset(DATA_DIR)
#
#    # Download pretrained vgg model
#    helper.maybe_download_pretrained_vgg(DATA_DIR)
#
#    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
#    # You'll need a GPU with at least 10 teraFLOPS to train on.
#    #  https://www.cityscapes-dataset.com/
#
#    with tf.Session() as sess:
#        # Path to vgg model
#        vgg_path = os.path.join(DATA_DIR, 'vgg')
#        # Create function to get batches
#        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)
#
#        # OPTIONAL: Augment Images for better results
#        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
#
#        # Build NN using load_vgg, layers, and optimize function
#        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
#        last_layer = layers(layer3_out, layer4_out, layer7_out, N_CLASSES)
#
#        correct_label = tf.placeholder(tf.int32)
#        learning_rate = tf.placeholder(tf.float32)
#        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, N_CLASSES)
#
#        # Train NN using the train_nn function
#        saver = tf.train.Saver(max_to_keep=3)
#        sess.run(tf.global_variables_initializer())
#        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input,
#                 correct_label, keep_prob, learning_rate, saver, LOG_DIR)
#
#        # Save inference data using helper.save_inference_samples
#        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, image_input)
#
#        # OPTIONAL: Apply the trained model to a video
        
        
def run():
    def make_dir(parent_dir, run_name):
        """
        Make directory under `parent_dir` (delete existing if already exists).
        """
        run_dir = parent_dir + os.path.sep + run_name
        if tf.gfile.Exists(run_dir):
            tf.gfile.DeleteRecursively(run_dir)
        tf.gfile.MakeDirs(run_dir)
        return run_dir

    run_dir = make_dir(RUNS_DIR, 'vgg_kitti_' + '{:.0f}'.format(time.time()))
    inference_dir = make_dir(run_dir, 'inference')

    imgs_train, imgs_val, gt_imgs_train, gt_imgs_val = helper.train_val_split(os.path.join(DATA_DIR, 'data_road/training'))    
    
    tf.reset_default_graph()

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create functions to get batches
        train_batches_fn = helper.gen_batch_function(imgs_train,
                                                     gt_imgs_train,
                                                     IMAGE_SHAPE,
                                                     augment=True)
        val_batches_fn = helper.gen_batch_function(imgs_val,
                                                   gt_imgs_val,
                                                   IMAGE_SHAPE)

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, N_CLASSES)

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, N_CLASSES)

        # Create summaries
        tf.summary.scalar('loss', cross_entropy_loss)

        # Create TensorBoard summaries
        summary_op = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(
            make_dir(run_dir, 'train'))
        val_writer = tf.summary.FileWriter(
            make_dir(run_dir, 'val'))

        saver = tf.train.Saver(max_to_keep=3)

        # Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, train_batches_fn,
                 train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, 
                 logits, 
                 summary_op, 
                 saver, 
                 run_dir,
                 train_writer, 
                 val_batches_fn, 
                 val_writer)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(inference_dir, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, image_input)

        
def inference_only(saved_model, inference_dir):
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(saved_model + '.meta')
        saver.restore(sess, saved_model)
        
        input_tensor_name = 'image_input:0'
        keep_prob_tensor_name = 'keep_prob:0'
        logits_tensor_name = 'Reshape:0'

        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name(input_tensor_name)
        keep_prob = graph.get_tensor_by_name(keep_prob_tensor_name)
        logits = graph.get_tensor_by_name(logits_tensor_name)
        
        # Path to vgg model
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(inference_dir, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, image_input)


if __name__ == '__main__':
    run()