# PnP-3D
This is a **tensorflow** implementation of PnP-3D module.

## RandLA-Net Usage
* Step 1. Create a linear mapping function ```def conv2d_simple``` in [helper_tf_util.py](https://github.com/QingyongHu/RandLA-Net/blob/master/helper_tf_util.py)
```
def conv2d_simple(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=False,
           stddev=1e-3,
           weight_decay=0.0):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        return outputs
```
* Step 2. Copy the functions ```def mish``` and ```def PnP3D``` in [pnp3d.py](https://github.com/ShiQiu0419/pnp-3d/blob/main/tensorflow/pnp3d.py), then put them inside of ```class Network``` in [RandLANet.py](https://github.com/QingyongHu/RandLA-Net/blob/master/RandLANet.py).
* Step 3. Put the following line behind each *dilated_res_block* in RandLA-Net (i.e., behind [this line](https://github.com/QingyongHu/RandLA-Net/blob/6b5445f5f279d33d2335e85ed39ca8b68cb1c57e/RandLANet.py#L115)):
```
f_encoder_i = self.pnp3d_module(f_encoder_i, inputs['xyz'][i], inputs['neigh_idx'][i], 'PnP3D_layer_' + str(i), is_training)
```

## SCF-Net Usage
* Step 1. Create a linear mapping function ```def conv2d_simple``` in [helper_tf_util.py](https://github.com/leofansq/SCF-Net/blob/main/helper_tf_util.py)
* Step 2. Copy the functions ```def mish``` and ```def PnP3D``` in [pnp3d.py](https://github.com/ShiQiu0419/pnp-3d/blob/main/tensorflow/pnp3d.py), then put them inside of ```class Network``` in [SCFNet.py](https://github.com/leofansq/SCF-Net/blob/main/SCFNet.py).
* Step 3. Put the following line behind each *scf_module* in SCF-Net (i.e., behind [this line](https://github.com/leofansq/SCF-Net/blob/a20343648594447ab5c31924f962cb0fc7bbd129/SCFNet.py#L117)):
```
f_encoder_i = self.pnp3d_module(f_encoder_i, inputs['xyz'][i], inputs['neigh_idx'][i], 'PnP3D_layer_' + str(i), is_training)
```
