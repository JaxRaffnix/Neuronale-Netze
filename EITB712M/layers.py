from builtins import range
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

    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)

    out = np.dot(x_reshaped, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)
    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout * (x > 0)

    return dx



def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HF and width WF.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HF, WF)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HF) / stride
      W' = 1 + (W + 2 * pad - WF) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    # Extract shapes and constants
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    H_out = 1+ (H +2 * pad - FH) // stride
    W_out = 1 + (W + 2 * pad - FW) // stride
    assert (H + 2 * pad - FH) % stride == 0, 'Non-integer H_out'
    assert (W + 2 * pad - FW) % stride == 0, 'Non-integer W_out'

    out = np.zeros((N, F, H_out, W_out))

    x_pad = np.pad(x, mode="constant", constant_values=0, pad_width=((0,0), (0,0), (pad,pad), (pad,pad)))

    for n in range(N):
        for f in range(F):
            for h in range(H_out):
                for wo in range(W_out):
                    h_start = h * stride
                    w_start = wo *stride
                    x_slice = x_pad[n, :, h_start:h_start+FH, w_start:w_start+FW]
                    out[n, f, h, wo] = np.sum(x_slice * w[f]) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
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
    # Extract shapes and constants
    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, Hpad, Wpad = x_pad.shape
    HF, WF = w.shape[2], w.shape[3]
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    pad = conv_param['pad']
    stride = conv_param['stride']
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))

    for n in range(N):
      for f in range(F):
          for h in range(outH):
              for wo in range(outW):
                  h_start = h * stride
                  w_start = wo *stride
                  x_slice = x_pad[n, :, h_start:h_start+HF, w_start:w_start+WF]    

                  dw[f] += x_slice * dout[n, f, h, wo]

                  dx_pad[n, :, h_start:h_start+HF, w_start:w_start+WF] += w[f] * dout[n, f, h, wo]

    dx = dx_pad[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_pad

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None

    # Extract shapes and constants
    N, C, H, W = x.shape
    HF = pool_param.get('pool_height', 2)
    WF = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    H_out = 1 + (H - HF) // stride
    W_out = 1 + (W - WF) // stride
    assert (H - HF) % stride == 0, 'Non-integer H_out'
    assert (W - WF) % stride == 0, 'Non-integer W_out'

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):           # batch
        for c in range(C):      
            for h in range(H_out):
                for wo in range(W_out):
                    h_start = h * stride
                    w_start = wo * stride
                    h_end = h_start + HF
                    w_end = w_start + WF

                    x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, h, wo] = np.max(x_slice)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    # Extract constants and shapes
    x, pool_param = cache
    N, C, H, W = x.shape
    HF = pool_param.get('pool_height', 2)
    WF = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    H_out = 1 + (H - HF) // stride
    W_out = 1 + (W - WF) // stride
    dx = np.zeros_like(x)

    for n in range(N):           # batch
        for c in range(C):      
            for h in range(H_out):
                for wo in range(W_out):
                  h_start = h * stride
                  w_start = wo * stride
                  h_end = h_start + HF
                  w_end = w_start + WF

                  x_slice = x[n, c, h_start:h_end, w_start:w_end]
                    
                  mask = (x_slice == np.max(x_slice))

                  dx[n, c, h_start:h_end, w_start:w_end] += mask * dout[n, c, h, wo]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
