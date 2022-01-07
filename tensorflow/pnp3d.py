#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Shi Qiu
@Contact: shi.qiu@anu.edu.au
@Time: 2021/01/06
"""
import tensorflow as tf
import numpy as np
import helper_tf_util

@staticmethod
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))

def pnp3d_module(self, feature, xyz, neigh_idx, name, is_training):
    d_out = feature.get_shape()[-1].value
    batch_size = tf.shape(feature)[0]
    num_points = tf.shape(feature)[1]

    # In PnP-3D, we only use half of the searched neighbors from original RandLA-Net
    neigh_idx = neigh_idx[:,:,:tf.shape(neigh_idx)[-1]//2]

    # Local Context fusion
    neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    xyz_concat = tf.concat([xyz_tile, relative_xyz], axis=-1)
    xyz_concat = helper_tf_util.conv2d(xyz_concat, d_out//2, [1, 1], name + 'mlp_xyz', [1, 1], 'VALID', True, is_training)

    neighbor_feat = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
    feat_tile = tf.tile(feature, [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_feat = feat_tile - neighbor_feat
    feat_concat = tf.concat([feat_tile, relative_feat], axis=-1)
    feat_concat = helper_tf_util.conv2d(feat_concat, d_out//2, [1, 1], name + 'mlp_feat', [1, 1], 'VALID', True, is_training)

    f_encoding = tf.concat([xyz_concat, feat_concat], axis=-1)
    f_encoding = tf.reduce_max(f_encoding, axis=2, keepdims=True)

    # Global Bilinear Regularization
    f_encoding_1 = helper_tf_util.conv2d_simple(f_encoding, d_out//8, [1, 1], name + 'conv_down_1', [1, 1], 'VALID')
    f_encoding_1 = tf.nn.relu(f_encoding_1)
    f_encoding_2 = helper_tf_util.conv2d_simple(f_encoding, d_out//8, [1, 1], name + 'conv_down_2', [1, 1], 'VALID')
    f_encoding_2 = tf.nn.relu(f_encoding_2)

    f_encoding_channel = tf.reduce_mean(f_encoding_1, axis=1)
    f_encoding_space = tf.reduce_mean(f_encoding_2, axis=-1)
    final_encoding = tf.matmul(f_encoding_space, f_encoding_channel)
    final_encoding = tf.sqrt(final_encoding+1e-12)
    final_encoding = tf.expand_dims(final_encoding, axis=2)

    final_encoding = final_encoding + f_encoding_1 + f_encoding_2
    final_encoding = helper_tf_util.conv2d(final_encoding, d_out, [1, 1], name + 'conv_up', [1, 1], 'VALID', True, is_training)

    # Mish Activation
    f_out = self.mish(f_encoding-final_encoding)

    return f_out
