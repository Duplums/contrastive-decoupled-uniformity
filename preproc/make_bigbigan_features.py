import io
import pickle
from pprint import pformat
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tqdm
import argparse

"""
    This script encodes ImageNet/ImageNet100 datasets using BigBiGAN's encoder.
    It requires TensorFlow >=2.8 to load BigBiGAN pre-trained model and PyTorch >= 1.6
    to re-order the encoded dataset. 
    It assumes you have manually downloaded ImageNet, see https://www.tensorflow.org/datasets/catalog/imagenet2012
    Pretrained model and notebook available here: https://tfhub.dev/s?publisher=deepmind&q=bigbigan 
"""


def print_signature(module):
    for signature in module.get_signature_names():
        print('Signature:', signature)
        print('Inputs:', pformat(module.get_input_info_dict(signature)))
        print('Outputs:', pformat(module.get_output_info_dict(signature)))

def load_ImageNet(ds_type, BASEDIR):
    [ds_train, ds_test], ds_info = tfds.load(ds_type, split=['train', 'validation'],
                                             data_dir=os.path.join(BASEDIR, "ImageNet"),
                                             download=True, shuffle_files=False,
                                             as_supervised=False, with_info=True,
                                             download_and_prepare_kwargs= {'download_dir':os.path.join(BASEDIR)})
    return [ds_train, ds_test], ds_info

# Take the center square crop of the image and resize to 256x256.
def crop_and_resize(d):
    d['image'] = tf.image.resize_bilinear([d['image']], [256, 256])[0]
    return d

# Convert images from [0, 255] uint8 to [-1, 1] float32.
def bytes_to_float(d):
    d['image'] = tf.cast(d['image'], tf.float32) / (255. / 2.) - 1
    return d


class BigBiGAN(object):

    def __init__(self, module):
        """Initialize a BigBiGAN from the given TF Hub module."""
        self._module = module

    def generate(self, z, upsample=False):
        """Run a batch of latents z through the generator to generate images.

        Args:
          z: A batch of 120D Gaussian latents, shape [N, 120].

        Returns: a batch of generated RGB images, shape [N, 128, 128, 3], range
          [-1, 1].
        """
        outputs = self._module(z, signature='generate', as_dict=True)
        return outputs['upsampled' if upsample else 'default']

    def make_generator_ph(self):
        """Creates a tf.placeholder with the dtype & shape of generator inputs."""
        info = self._module.get_input_info_dict('generate')['z']
        return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

    def gen_pairs_for_disc(self, z):
        """Compute generator input pairs (G(z), z) for discriminator, given z.

        Args:
          z: A batch of latents (120D standard Gaussians), shape [N, 120].

        Returns: a tuple (G(z), z) of discriminator inputs.
        """
        # Downsample 256x256 image x for 128x128 discriminator input.
        x = self.generate(z)
        return x, z

    def encode(self, x, return_all_features=False):
        """Run a batch of images x through the encoder.

        Args:
          x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
            [-1, 1].
          return_all_features: If True, return all features computed by the encoder.
            Otherwise (default) just return a sample z_hat.

        Returns: the sample z_hat of shape [N, 120] (or a dict of all features if
          return_all_features).
        """
        outputs = self._module(x, signature='encode', as_dict=True)
        return outputs if return_all_features else outputs['z_sample']

    def make_encoder_ph(self):
        """Creates a tf.placeholder with the dtype & shape of encoder inputs."""
        info = self._module.get_input_info_dict('encode')['x']
        return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

    def enc_pairs_for_disc(self, x):
        """Compute encoder input pairs (x, E(x)) for discriminator, given x.

        Args:
          x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
            [-1, 1].

        Returns: a tuple (downsample(x), E(x)) of discriminator inputs.
        """
        # Downsample 256x256 image x for 128x128 discriminator input.
        x_down = tf.nn.avg_pool(x, ksize=2, strides=2, padding='SAME')
        z = self.encode(x)
        return x_down, z

def extract_features(tf_dataset: tf.data.ImageNet,
                     name: str,
                     batch_size: int=128):
    features = []
    y_true = []
    filenames = []
    bar = tqdm.tqdm(desc=name, total=int(tf_dataset.cardinality().eval(session=sess)/float(batch_size)))
    for d in tfds.as_numpy(ds.batch(batch_size)):
        images = d['image']
        labels = d['label']
        file_names = d['file_name']
        y_true.extend(labels)
        filenames.extend(file_names)
        with tf.device('/device:GPU:0'):
            _out_features = sess.run(enc_features, feed_dict={enc_ph: images})
            features.extend(_out_features["bn_crelu_feat"])
        bar.update()
    return features, y_true, filenames


def reorder_extracted_features(features, y_true,
                               filenames,
                               torch_dataset):
    # From ImageNet encoded with BigBiGAN in TensorFlow,
    # order the data according to PyTorch ImageNet dataset's ordering

    features = np.array(features, dtype=np.float32)
    # Store the ordered path in pandas dataframe + preproc
    pths_left = pd.DataFrame(torch_dataset.samples, columns=["filenames", "target"])
    pths_left["filenames"] = pths_left.filenames.apply(lambda pth: os.path.basename(pth).encode("UTF-8"))
    # Merge these paths with the ones in meta, preserving the right index column
    pths_right = pd.DataFrame(dict(y_true=np.array(y_true),
                                   filenames=np.array(filenames))).reset_index()
    print("Merging indexes...", flush=True)
    pths_merged = pd.merge(pths_left, pths_right, on="filenames", how="inner", validate="1:1")
    assert len(pths_merged) == len(pths_left)
    print("Merge successful !", flush=True)
    if len(pths_right) == len(pths_left):
        assert (pths_merged.y_true == pths_merged.target).all(), "Labels inconsistent"
    # Extract the index for masking
    mask = pths_merged["index"].values
    all_labels = pths_merged["target"].values
    print("Re-ordering features...", flush=True)
    all_features = features[mask]
    print("Re-ordering successful !", flush=True)
    return all_features, all_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True, help="path to data.")
    parser.add_argument("--db", type=str, required=True, default="imagenet100",
                        choices=["imagenet100"], help="dataset to encode.")
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    root = args.root
    dataset =  args.db
    module_path = 'https://tfhub.dev/deepmind/bigbigan-resnet50/1'
    module = hub.Module(module_path)  # inference
    print_signature(module)

    # Load BigBiGAN pre-trained model
    bigbigan = BigBiGAN(module)
    enc_ph = bigbigan.make_encoder_ph()
    # Encode features
    enc_features = bigbigan.encode(enc_ph, return_all_features=True)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    (ds_train, ds_test), ds_info = load_ImageNet("imagenet2012", root)

    # Preprocess the images
    ds_train = ds_train.map(crop_and_resize)
    ds_test = ds_test.map(crop_and_resize)
    ds_train = ds_train.map(bytes_to_float)
    ds_test = ds_test.map(bytes_to_float)

    for ds, ds_name in zip([ds_test, ds_train], ["val", "train"]):
        # Encode ImageNet with BigBiGAN
        features, y_true, filenames = extract_features(ds, ds_name, batch_size=args.batch_size)

        # Re-order features
        from datasets.imagenet100 import ImageNet100
        if args.db == "imagenet100":
            torch_dataset = ImageNet100(root, split=ds_name)
        else:
            raise NotImplementedError()
        prior, labels = reorder_extracted_features(features, y_true, filenames, torch_dataset)

        # Dump features
        np.savez(os.path.splitext(torch_dataset.prior_path)[0], prior=prior, labels=labels)




