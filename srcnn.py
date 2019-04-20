import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from skimage import measure


def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    image = image / 255.
    label_ = label_ / 255.

    input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
}

biases = {
    'b1': tf.Variable(tf.zeros([64]), name='b1'),
    'b2': tf.Variable(tf.zeros([32]), name='b2'),
    'b3': tf.Variable(tf.zeros([1]), name='b3')
}

"""Define the model layers with three convolutional layers
"""
## ------ Add your code here: to compute feature maps of input low-resolution images
# replace 'None' with your layers: use the tf.nn.conv2d() and tf.nn.relu()
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs, filter=weights['w1'], strides=[1, 1, 1, 1], padding='VALID', name='conv1') + biases['b1'])
##------ Add your code here: to compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter=weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])
##------ Add your code here: compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, filter=weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']

"""Load the pre-trained model file
"""
model_path = './model/model.npy'
model = np.load(model_path, encoding='latin1').item()

##------ Add your code here: show the weights of model and try to visualisa
# variabiles (w1, w2, w3)
weight_w1 = model['w1']
weight_w2 = model['w2']
weight_w3 = model['w3']

biase_b1 = model['b1']
biase_b2 = model['b2']
biase_b3 = model['b3']

def vis_w1():
  fig = plt.figure(figsize=(8, 8))

  for i in range(64):
    suba = fig.add_subplot(8, 8, i + 1)
    suba.imshow(weight_w1[:, :, 0, i], cmap='gray')
    plt.axis('off')
  plt.show()


def vis_w2():
  fig = plt.figure(figsize=(8, 8))

  for i in range(32):
    suba = fig.add_subplot(4, 8, i + 1)
    suba.imshow(weight_w2[:, :, 0, i], cmap='gray')
    plt.axis('off')
  plt.show()


def vis_w3():
  fig = plt.figure(figsize=(2, 2))
  plt.imshow(weight_w1[:, :, 0, 0], cmap='gray')
  plt.show()

"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
    sess.run(weights[key].assign(model[key]))

for key in biases.keys():
    sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis=0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output0 = sess.run(conv1, feed_dict={inputs: input_})
ouput_ = sess.run(conv3, feed_dict={inputs: input_})

##------ Add your code here: save the blurred and SR images and compute the psnr
# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'
grdTru = scipy.misc.imsave('grdTru.jpg', groudtruth_image)
blur = scipy.misc.imsave('blur.jpg', blurred_image)

output1 = ouput_.astype(np.float64)
#print(output1.shape)
outHighRes = scipy.misc.imsave('outHighRes.jpg', output1[0,:,:,0])

bicPSNR = measure.compare_psnr(groudtruth_image, blurred_image)
supPSNR = measure.compare_psnr(groudtruth_image[6:249,6:249], output1[0, :, :, 0])

print('PSNR of Bicubic: ', bicPSNR)
print('PSNR of SupRes: ', supPSNR)