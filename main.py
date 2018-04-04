import os.path
import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

LEARNING_RATE = 0.001
DECAY_RATE = 0.90
DECAY_AFTER_N_STEPS = 50
KEEP_PROB = 0.5
EPOCHS = 50
BATCH_SIZE = 5
CLASSES = 2
IMAGE_SHAPE = (160, 576)
L2_REG = 0.001
STD_DEV = 0.01
DATA_DIR = './data'
MODEL_DIR = './model'
VIDEO_DIR = './video'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (input_image, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name) # pooled layer 3
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name) # pooled layer 4
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name) # convolved layer 7
    
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # layer 7 1x1 convolution to conserve spatial information
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    # layer 7 1x1 convolution upsampled to reverse the convolution operation
    layer7_upsampled = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    # layer 4 1x1 convolution to conserve spatial information
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    # Skip connection between convolved layer 4 & convolved + upsampled layer 7 to retain the original context
    layer4_7_skip_connection = tf.add(layer7_upsampled, layer4_1x1)
    
    # Upscaling again in preparation for creating the skip connection between layer 7 & layer 3
    layer7_final = tf.layers.conv2d_transpose(layer4_7_skip_connection, num_classes, 4, 2, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    # layer 3 1x1 convolution to conserve spatial information
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))
    
    # Skip connection between convolved layer 3 & upsampled layer 7 
    layer3_7_skip_connection = tf.add(layer3_1x1, layer7_final)
    
    # final layer upscaling
    layer_last = tf.layers.conv2d_transpose(layer3_7_skip_connection, num_classes, 16, 8, padding='same',
                             kernel_initializer= tf.random_normal_initializer(stddev=STD_DEV),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG))

    return layer_last
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss, decaying_learning_rate)
    """
    # Reshape data so that there are n rows & 2 columns (1 column for each label)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # Apply decaying learning rate i.e. the learning rate decreases as the epochs increase
    global_step = tf.Variable(0, trainable=False) 
    initial_learning_rate = learning_rate
    # 1 step = 1 training batch
    decaying_learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, DECAY_AFTER_N_STEPS, DECAY_RATE, staircase=True)
    
    # Optimizer for reducing loss
    optimizer = tf.train.AdamOptimizer(decaying_learning_rate)
    
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)
    
    return logits, train_op, cross_entropy_loss, decaying_learning_rate
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, decaying_learning_rate):
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
    :param decaying_learning_rate: TF Placeholder for decaying learning rate
    :return: Dictionary of (epoch_no, avg_loss_across_batches)
    """
    epoch_loss = {}
    batch_loss = []
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        start_time = time.time()
        print('Epoch: {} START...'.format(epoch + 1))
        batch_counter = 1
        for image, label in get_batches_fn(batch_size):
            _, loss, decaying_rate = sess.run([train_op, cross_entropy_loss, decaying_learning_rate],
                                              feed_dict={
                                                      input_image: image,
                                                      correct_label: label,
                                                      keep_prob: KEEP_PROB,
                                                      learning_rate: LEARNING_RATE
                                                      })  
            print("  Batch {} >>> Loss = {:.4f}, Learning Rate = {:.6f}".format(batch_counter, loss, decaying_rate))
            batch_loss.append(loss)
            batch_counter += 1
        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed//3600
        minutes = (elapsed%3600)//60
        seconds = (elapsed%3600)%60
        print("Epoch: {} END. Time taken: {:.0f} hours {:.0f} minutes {:.0f} seconds\n".format(epoch + 1, hours, minutes, seconds))
        epoch_loss[epoch] = np.average(batch_loss)
    return epoch_loss
    
tests.test_train_nn(train_nn)

def run():
    num_classes = CLASSES
    image_shape = IMAGE_SHAPE
    data_dir = DATA_DIR
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss, decaying_learning_rate = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        start_time = time.time()
        
        epoch_loss = train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, 
                              cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, decaying_learning_rate)
        
        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed//3600
        minutes = (elapsed%3600)//60
        seconds = (elapsed%3600)%60
        print("Training time: {:.0f} hours {:.0f} minutes {:.0f} seconds".format(hours, minutes, seconds))     
        
        log_file_path = './' + str(EPOCHS) + '_log.txt'
        log_file = open(log_file_path, 'w') 
        log_file.write('Epoch,Loss\n')
        for key in epoch_loss.keys():
            log_file.write('{},{}\n'.format(key, epoch_loss[key]))
        log_file.close()
        
        start_time = time.time()
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed//3600
        minutes = (elapsed%3600)//60
        seconds = (elapsed%3600)%60
        print("Inference time: {:.0f} hours {:.0f} minutes {:.0f} seconds".format(hours, minutes, seconds))
        
        # Save model for later use
        #saver = tf.train.Saver()
        #saver.save(sess, MODEL_DIR + '/model_meta')
        #print("Model saved to {} directory".format(MODEL_DIR))
        
        # OPTIONAL: Apply the trained model to a video
        start_time = time.time()
        processed_frames = []
        # Load video 
        #video_clip = VideoFileClip(VIDEO_DIR + '/project_video.mp4')
        video_clip = VideoFileClip(VIDEO_DIR + '/solidWhiteRight.mp4')
        frame_counter = 1;
        for frame in video_clip.iter_frames():
            processed_frame = helper.segment_single_image(sess, logits, keep_prob, input_image, frame, image_shape)
            # Collect processed frame
            processed_frames.append(processed_frame)
            print("Frame {} processed".format(frame_counter))
            frame_counter += 1 
        # Stitcha all frames to get the video
        processed_video = ImageSequenceClip(processed_frames, fps=video_clip.fps)
        processed_video.write_videofile(VIDEO_DIR + '/solidWhiteRight_processed.mp4', audio=False)
        print("Processed video written to {} directory".format(VIDEO_DIR))
        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed//3600
        minutes = (elapsed%3600)//60
        seconds = (elapsed%3600)%60
        print("Video processing time: {:.0f} hours {:.0f} minutes {:.0f} seconds".format(hours, minutes, seconds))


if __name__ == '__main__':
    run()

