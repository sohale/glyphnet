# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import tensorflow as tf

# putputs
global weights_count

def create_weight_variable(DTYPE):
    global weights_count
    randinitval = tf.random.uniform([1], -1, 1, seed=0)  # doesn accept `trainable=False,`
    w1 = tf.Variable(initial_value=randinitval, trainable=True, dtype=DTYPE)
    weights_count += 1
    return w1

def create_bias_variable(DTYPE):
    return tf.Variable(initial_value=0.0, trainable=True, dtype=DTYPE)

def coord_to_str(coord):
    ndim = len(coord)
    if ndim == 2:
        (x,y) = coord
        return "x%dy%d"%(x,y)
    elif ndim == 1:
        return "i%d"%(coord[0])
    elif ndim == 3:
        (x,y, rgb_ch) = coord
        return "x%dy%dc%d"%(x,y,rgb_ch)
    elif ndim > 3:
        s = ""
        for x in coord:
            s += "i%d"% x
        return s
    else:
        raise Exception('incorrect number of virtual coordinates dims')

"""
    #prune_rule(coords_prev, coords_next) -> bool
    no dependence to MLNTopology
"""
def wireup_layer(next_shape_int, connectivity, coords_next, prev_layer_outputs, coords_prev, prune_rule, nonlinearity1, weight_dtype, bias_dtype, name):
    assert isinstance(next_shape_int, int)
    with tf.name_scope('L' + name):
        ll = []
        for i_next in range(next_shape_int):
            coord_next = coords_next[i_next]
            cuname1 = coord_to_str(coord_next)
            with tf.name_scope(cuname1):
                suminp = 0.0
                # todo: check if correctly fixed undefined variable prev_shape_int: (unverified)
                prev_shape_int = len(coords_prev)
                # prev_layer_count = prev_shape_int
                for j_prev in range(prev_shape_int):
                    coord_prev = coords_prev[j_prev]
                    # apply synaptic prune rule for connectivity:
                    # not??
                    if not prune_rule(coord_prev, coord_next):
                        v1 = prev_layer_outputs[:, j_prev]
                        w1 = create_weight_variable(weight_dtype)
                        suminp = suminp + w1 * v1
                b1 = create_bias_variable(bias_dtype)
                suminp = suminp + b1
                # convolution unit output
                conv_unit_outp = nonlinearity1( suminp )

                ll += [conv_unit_outp[:, None, :]] # prepare for row-like structure

        layer_h1 = tf.concat(ll, axis=1) # row: (W*H) x RGB3

        newshape = (next_shape_int,)
        NEWRESHAPE = [-1] + [d for d in newshape]
        reshaped_hidden_layer = tf.reshape(layer_h1, NEWRESHAPE, name=name)

    return reshaped_hidden_layer


PIXEL_DTYPE = tf.float32
HL_DTYPE = tf.float32
WEIGHT_DTYPE = tf.float32

def wireup(topology):
    # Placeholder for actual implementation
    # Never tested:

    out = \
    wireup_layer(
        next_shape_int,
        connectivity,
        coords_next,
        prev_layer_outputs,
        coords_prev,
        prune_rule,
        nonlinearity1,
        WEIGHT_DTYPE,
        bias_dtype=HL_DTYPE,
        name="testlayer1")

    return out

# self test?
def test_em_all():
    topology = ...
    wireup(topology)
    print('main unittest ended peacefully.')

    pass

if __name__ == '__main__':
    test_em_all()
