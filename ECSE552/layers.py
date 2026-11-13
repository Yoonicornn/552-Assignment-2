import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    #############################################################################
    nelements = w.shape[1]
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.dot(dout.T, x.reshape(x.shape[0], -1)).T
    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T).reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    # #############################################################################
    # # TODO: Implement the convolutional forward pass.                           #
    # # Hint: you can use the function np.pad for padding.                        #
    # # Hint: You can also use im2col or im2col_indices see the file im2col.py    #
    # #       for further information                                             #
    # #############################################################################
    # pass
    # #############################################################################
    # #                             END OF YOUR CODE                              #
    # #############################################################################

    # input dims: N samples, C channels, H x W image
    N, C, H, W = x.shape
    # filter dims: F filters, each C x HH x WW
    F, _, HH, WW = w.shape
    # read stride and padding
    stride = conv_param['stride']
    pad = conv_param['pad']
    # Eqn 2.0: output height and width
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    # zero-pad only H and W axes, (N,C,H,W) -> (N,C,H+2p,W+2p)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # allocate output, (N,F,H_out,W_out)
    out = np.zeros((N, F, H_out, W_out))
    # loop over samples
    for n in range(N):
        # loop over filters
        for f in range(F):
            # loop over output rows
            for i in range(H_out):
                # loop over output columns
                for j in range(W_out):
                    # top-left corner of receptive field
                    hs = i * stride
                    ws = j * stride
                    # receptive field, (C,HH,WW)
                    window = x_pad[n, :, hs:hs + HH, ws:ws + WW]
                    # Eqn 2.1: elementwise product with filter, sum, add bias -> scalar
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    # #############################################################################
    # # TODO: Implement the convolutional backward pass.                          #
    # #############################################################################
    # pass
    # #############################################################################
    # #                             END OF YOUR CODE                              #
    # #############################################################################
    
    # unpack cache
    x, w, b, conv_param = cache
    # input dims
    N, C, H, W = x.shape
    # filter dims
    F, _, HH, WW = w.shape
    # stride and padding
    stride = conv_param['stride']
    pad = conv_param['pad']
    # output spatial dims, (N,F,H_out,W_out)
    _, _, H_out, W_out = dout.shape
    # padded input (same as forward), (N,C,H+2p,W+2p)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # gradient w.r.t. padded input, (N,C,H+2p,W+2p)
    dx_pad = np.zeros_like(x_pad)
    # gradient w.r.t. weights, (F,C,HH,WW)
    dw = np.zeros_like(w)
    # gradient w.r.t. bias, (F,)
    db = np.zeros_like(b)
    # loop over samples
    for n in range(N):
        # loop over filters
        for f in range(F):
            # loop over output rows
            for i in range(H_out):
                # loop over output columns
                for j in range(W_out):
                    # top-left corner of receptive field
                    hs = i * stride
                    ws = j * stride
                    # upstream gradient of this output, scalar
                    g = dout[n, f, i, j]
                    # receptive field used in forward, (C,HH,WW)
                    window = x_pad[n, :, hs:hs + HH, ws:ws + WW]
                    # Eqn 2.2: bias gets the upstream gradient
                    db[f] += g
                    # Eqn 2.3: weights get window * upstream, (C,HH,WW)
                    dw[f] += g * window
                    # Eqn 2.4: input window gets filter * upstream, (C,HH,WW)
                    dx_pad[n, :, hs:hs + HH, ws:ws + WW] += g * w[f]
    # Eqn 2.5: remove padding, (N,C,H+2p,W+2p) -> (N,C,H,W)
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    # #############################################################################
    # # TODO: Implement the max pooling forward pass                              #
    # #############################################################################
    # pass
    # #############################################################################
    # #                             END OF YOUR CODE                              #
    # #############################################################################

    # input dims
    N, C, H, W = x.shape
    # pooling window size and stride
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']
    # Eqn 3.0: output spatial size
    H_out = 1 + (H - ph) // stride
    W_out = 1 + (W - pw) // stride
    # allocate output, (N,C,H_out,W_out)
    out = np.zeros((N, C, H_out, W_out))
    # loop over samples
    for n in range(N):
        # loop over channels (pooling is per channel)
        for c in range(C):
            # loop over output rows
            for i in range(H_out):
                # loop over output columns
                for j in range(W_out):
                    # top-left corner of the pooling window
                    hs = i * stride
                    ws = j * stride
                    # Eqn 3.1: max over window (ph,pw) -> scalar
                    out[n, c, i, j] = np.max(x[n, c, hs:hs + ph, ws:ws + pw])
    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    # #############################################################################
    # # TODO: Implement the max pooling backward pass                             #
    # #############################################################################
    # pass
    # #############################################################################
    # #                             END OF YOUR CODE                              #
    # #############################################################################

    # unpack cache
    x, pool_param = cache
    # input dims
    N, C, H, W = x.shape
    # pooling window size and stride
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    stride = pool_param['stride']
    # output spatial dims
    _, _, H_out, W_out = dout.shape
    # gradient w.r.t. input, (N,C,H,W)
    dx = np.zeros_like(x)
    # loop over samples
    for n in range(N):
        # loop over channels
        for c in range(C):
            # loop over output rows
            for i in range(H_out):
                # loop over output columns
                for j in range(W_out):
                    # top-left corner of the window
                    hs = i * stride
                    ws = j * stride
                    # pooling window, (ph,pw)
                    window = x[n, c, hs:hs + ph, ws:ws + pw]
                    # Eqn 3.2: (row, col) of the max inside the window
                    r, s = np.unravel_index(np.argmax(window), window.shape)
                    # Eqn 3.3: route upstream gradient only to the max position
                    dx[n, c, hs + r, ws + s] += dout[n, c, i, j]

    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    # if mode == 'train':
    #     #############################################################################
    #     # TODO: Implement the training-time forward pass for batch normalization.   #
    #     # Use minibatch statistics to compute the mean and variance, use these      #
    #     # statistics to normalize the incoming data, and scale and shift the        #
    #     # normalized data using gamma and beta.                                     #
    #     #                                                                           #
    #     # You should store the output in the variable out. Any intermediates that   #
    #     # you need for the backward pass should be stored in the cache variable.    #
    #     #                                                                           #
    #     # You should also use your computed sample mean and variance together with  #
    #     # the momentum variable to update the running mean and running variance,    #
    #     # storing your result in the running_mean and running_var variables.        #
    #     #############################################################################
    #     pass
    #     #############################################################################
    #     #                             END OF YOUR CODE                              #
    #     #############################################################################
    # elif mode == 'test':
    #     #############################################################################
    #     # TODO: Implement the test-time forward pass for batch normalization. Use   #
    #     # the running mean and variance to normalize the incoming data, then scale  #
    #     # and shift the normalized data using gamma and beta. Store the result in   #
    #     # the out variable.                                                         #
    #     #############################################################################
    #     pass
    #     #############################################################################
    #     #                             END OF YOUR CODE                              #
    #     #############################################################################

    if mode == 'train':
        # Eqn 1.0: per-feature mean over the batch, (N,D) -> (D,)
        sample_mean = np.mean(x, axis=0)
        # Eqn 1.1: center the data, (N,D) - (D,) broadcast -> (N,D)
        xmu = x - sample_mean
        # Eqn 1.2: per-feature (biased) variance, (N,D) -> (D,)
        sample_var = np.mean(xmu ** 2, axis=0)
        # Eqn 1.3: standard deviation with eps for stability, (D,)
        std = np.sqrt(sample_var + eps)
        # Eqn 1.4: inverse std, (D,)
        istd = 1.0 / std
        # Eqn 1.5: normalized input, (N,D) * (D,) -> (N,D)
        xhat = xmu * istd
        # Eqn 1.6: scale and shift, (D,)*(N,D) + (D,) -> (N,D)
        out = gamma * xhat + beta
        # update running mean with momentum, (D,)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        # update running variance with momentum, (D,)
        running_var = momentum * running_var + (1 - momentum) * sample_var
        # save intermediates for backward pass
        cache = (xhat, xmu, istd, gamma)
    elif mode == 'test':
        # normalize with running statistics, (N,D) - (D,) / (D,) -> (N,D)
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        # scale and shift, (N,D)
        out = gamma * xhat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    # #############################################################################
    # # TODO: Implement the backward pass for batch normalization. Store the      #
    # # results in the dx, dgamma, and dbeta variables.                           #
    # #############################################################################
    # pass
    # #############################################################################
    # #                             END OF YOUR CODE                              #
    # #############################################################################

    # unpack values saved in forward pass
    xhat, xmu, istd, gamma = cache
    # batch size N and feature size D
    N, D = dout.shape
    # Eqn 1.7: dbeta = sum over batch of dout, (N,D) -> (D,)
    dbeta = np.sum(dout, axis=0)
    # Eqn 1.8: dgamma = sum over batch of dout * xhat, (N,D) -> (D,)
    dgamma = np.sum(dout * xhat, axis=0)
    # Eqn 1.9: gradient into xhat, (N,D) * (D,) -> (N,D)
    dxhat = dout * gamma
    # Eqn 1.10: branch 1 of xmu (through xhat = xmu * istd), (N,D)
    dxmu1 = dxhat * istd
    # Eqn 1.11: gradient into istd (summed over batch), (N,D) -> (D,)
    distd = np.sum(dxhat * xmu, axis=0)
    # Eqn 1.12: istd = 1/std -> d/dstd = -1/std^2, (D,)
    dstd = distd * (-istd ** 2)
    # Eqn 1.13: std = sqrt(var+eps) -> d/dvar = 0.5/std, (D,)
    dvar = dstd * 0.5 * istd
    # Eqn 1.14: branch 2 of xmu (through var = mean(xmu^2)), (N,D)
    dxmu2 = (2.0 / N) * xmu * dvar
    # Eqn 1.15: sum both branches into xmu, (N,D)
    dx1 = dxmu1 + dxmu2
    # Eqn 1.16: gradient into mu (xmu = x - mu), (N,D) -> (D,)
    dmu = -np.sum(dx1, axis=0)
    # Eqn 1.17: x gets direct branch + mean branch (mu = mean(x)), (N,D)
    dx = dx1 + dmu / N

    return dx, dgamma, dbeta