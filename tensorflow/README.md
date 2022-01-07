# PnP-3D for RandLA-Net

## Usage
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
* Step 2. In the network file, [RandLANet.py](https://github.com/QingyongHu/RandLA-Net/blob/master/RandLANet.py), create a mish function as:
```
def mish(x):
    return x*tf.math.tanh(tf.math.softplus(x))
```
and create PnP-3D module as:
```

```
