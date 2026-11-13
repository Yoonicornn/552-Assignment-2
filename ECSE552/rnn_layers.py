import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  # ##############################################################################
  # # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # # hidden state and any values you need for the backward pass in the next_h   #
  # # and cache variables respectively.                                          #
  # ##############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # Eqn 4.0: pre-activation, (N,D)(D,H) + (N,H)(H,H) + (H,) -> (N,H)
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  # Eqn 4.1: tanh activation, (N,H)
  next_h = np.tanh(a)
  # save values for backward pass
  cache = (x, prev_h, Wx, Wh, next_h)
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  # ##############################################################################
  # # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  # #                                                                            #
  # # HINT: For the tanh function, you can compute the local derivative in terms #
  # # of the output value from tanh.                                             #
  # ##############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # unpack cache
  x, prev_h, Wx, Wh, next_h = cache
  # Eqn 4.2: through tanh, d tanh(a)/da = 1 - tanh(a)^2, (N,H)
  da = dnext_h * (1 - next_h ** 2)
  # Eqn 4.3: gradient w.r.t. x, (N,H)(H,D) -> (N,D)
  dx = np.dot(da, Wx.T)
  # Eqn 4.4: gradient w.r.t. prev_h, (N,H)(H,H) -> (N,H)
  dprev_h = np.dot(da, Wh.T)
  # Eqn 4.5: gradient w.r.t. Wx, (D,N)(N,H) -> (D,H)
  dWx = np.dot(x.T, da)
  # Eqn 4.6: gradient w.r.t. Wh, (H,N)(N,H) -> (H,H)
  dWh = np.dot(prev_h.T, da)
  # Eqn 4.7: gradient w.r.t. b, sum over batch, (N,H) -> (H,)
  db = np.sum(da, axis=0)

  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  # ##############################################################################
  # # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # # input data. You should use the rnn_step_forward function that you defined  #
  # # above.                                                                     #
  # ##############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # input dims: N sequences, T timesteps, D features
  N, T, D = x.shape
  # hidden size
  H = h0.shape[1]
  # store all hidden states, (N,T,H)
  h = np.zeros((N, T, H))
  # hidden state at t = 0, (N,H)
  prev_h = h0
  # list of per-timestep caches
  step_caches = []
  # loop forward in time
  for t in range(T):
    # one RNN step: x[:,t,:] is (N,D), prev_h is (N,H) -> next_h (N,H)
    next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
    # save hidden state of timestep t
    h[:, t, :] = next_h
    # save step cache for backward
    step_caches.append(step_cache)
    # current state becomes previous state for next step
    prev_h = next_h
  # cache: step caches + input shape
  cache = (step_caches, x.shape)

  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  # ##############################################################################
  # # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # # sequence of data. You should use the rnn_step_backward function that you   #
  # # defined above.                                                             #
  # ##############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # unpack step caches and input shape
  step_caches, (N, T, D) = cache
  # hidden size
  H = dh.shape[2]
  # gradient w.r.t. inputs, (N,T,D)
  dx = np.zeros((N, T, D))
  # accumulators for shared weights, (D,H), (H,H), (H,)
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)
  # gradient coming from the future step (zero after last step), (N,H)
  dprev_h = np.zeros((N, H))
  # loop backward in time
  for t in reversed(range(T)):
    # Eqn 4.8: total grad on h_t = loss grad at t + grad from step t+1, (N,H)
    dnext_h = dh[:, t, :] + dprev_h
    # Eqn 4.9: backprop one step
    dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, step_caches[t])
    # store input gradient of timestep t, (N,D)
    dx[:, t, :] = dx_t
    # Eqn 4.10: weights shared over time -> sum their gradients
    dWx += dWx_t
    dWh += dWh_t
    db += db_t
  # Eqn 4.11: gradient left after t = 0 goes to h0, (N,H)
  dh0 = dprev_h

  return dx, dh0, dWx, dWh, db

def sigmoid(z):
  """Numerically stable sigmoid. ADDED by me, since the original file missed this function"""
  # exp of a non-positive number never overflows
  e = np.exp(-np.abs(z))
  # z>=0: 1/(1+e^-z) ; z<0: e^z/(1+e^z)
  return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  # #############################################################################
  # # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # # You may want to use the numerically stable sigmoid implementation above.  #
  # #############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # hidden size
  H = prev_h.shape[1]
  # Eqn 5.0: activation vector, (N,D)(D,4H) + (N,H)(H,4H) + (4H,) -> (N,4H)
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  # Eqn 5.1: split a into 4 blocks of size H, each (N,H)
  ai = a[:, 0:H]
  af = a[:, H:2 * H]
  ao = a[:, 2 * H:3 * H]
  ag = a[:, 3 * H:4 * H]
  # Eqn 5.2: input gate, (N,H)
  i = sigmoid(ai)
  # Eqn 5.3: forget gate, (N,H)
  f = sigmoid(af)
  # Eqn 5.4: output gate, (N,H)
  o = sigmoid(ao)
  # Eqn 5.5: block input, (N,H)
  g = np.tanh(ag)
  # Eqn 5.6: next cell state, (N,H)
  next_c = f * prev_c + i * g
  # tanh of new cell state (reused in backward), (N,H)
  tanh_c = np.tanh(next_c)
  # Eqn 5.7: next hidden state, (N,H)
  next_h = o * tanh_c
  # save values for backward pass
  cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, tanh_c)
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  # #############################################################################
  # # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  # #                                                                           #
  # # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # # the output value from the nonlinearity.                                   #
  # #############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # unpack cache
  x, prev_h, prev_c, Wx, Wh, i, f, o, g, tanh_c = cache
  # Eqn 5.8: h = o*tanh(c) -> grad on o, (N,H)
  do = dnext_h * tanh_c
  # Eqn 5.9: total grad on c_t = from next step + through h = o*tanh(c), (N,H)
  dc = dnext_c + dnext_h * o * (1 - tanh_c ** 2)
  # Eqn 5.10: c = f*prev_c + i*g -> grad on f, (N,H)
  df = dc * prev_c
  # Eqn 5.11: grad on prev_c, (N,H)
  dprev_c = dc * f
  # Eqn 5.12: grad on i, (N,H)
  di = dc * g
  # Eqn 5.13: grad on g, (N,H)
  dg = dc * i
  # Eqn 5.14: through sigmoid, sigma' = sigma(1-sigma), (N,H)
  dai = di * i * (1 - i)
  daf = df * f * (1 - f)
  dao = do * o * (1 - o)
  # Eqn 5.15: through tanh, tanh' = 1 - tanh^2, (N,H)
  dag = dg * (1 - g ** 2)
  # Eqn 5.16: stack in same order as split, 4 x (N,H) -> (N,4H)
  da = np.hstack((dai, daf, dao, dag))
  # Eqn 5.17: gradient w.r.t. x, (N,4H)(4H,D) -> (N,D)
  dx = np.dot(da, Wx.T)
  # Eqn 5.18: gradient w.r.t. prev_h, (N,4H)(4H,H) -> (N,H)
  dprev_h = np.dot(da, Wh.T)
  # Eqn 5.19: gradient w.r.t. Wx, (D,N)(N,4H) -> (D,4H)
  dWx = np.dot(x.T, da)
  # Eqn 5.20: gradient w.r.t. Wh, (H,N)(N,4H) -> (H,4H)
  dWh = np.dot(prev_h.T, da)
  # Eqn 5.21: gradient w.r.t. b, (N,4H) -> (4H,)
  db = np.sum(da, axis=0)

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  # #############################################################################
  # # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # # You should use the lstm_step_forward function that you just defined.      #
  # #############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # input dims
  N, T, D = x.shape
  # hidden size
  H = h0.shape[1]
  # store all hidden states, (N,T,H)
  h = np.zeros((N, T, H))
  # initial hidden state, (N,H)
  prev_h = h0
  # initial cell state is zero, (N,H)
  prev_c = np.zeros((N, H))
  # list of per-timestep caches
  step_caches = []
  # loop forward in time
  for t in range(T):
    # one LSTM step: x[:,t,:] (N,D) -> new h and c, both (N,H)
    prev_h, prev_c, step_cache = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
    # save hidden state of timestep t
    h[:, t, :] = prev_h
    # save step cache
    step_caches.append(step_cache)
  # cache: step caches + input shape
  cache = (step_caches, x.shape)

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  # #############################################################################
  # # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # # You should use the lstm_step_backward function that you just defined.     #
  # #############################################################################
  # pass
  # ##############################################################################
  # #                               END OF YOUR CODE                             #
  # ##############################################################################

  # unpack step caches and input shape
  step_caches, (N, T, D) = cache
  # hidden size
  H = dh.shape[2]
  # gradient w.r.t. inputs, (N,T,D)
  dx = np.zeros((N, T, D))
  # accumulators for shared weights, (D,4H), (H,4H), (4H,)
  dWx = np.zeros((D, 4 * H))
  dWh = np.zeros((H, 4 * H))
  db = np.zeros(4 * H)
  # gradients flowing back from step t+1 (zero after last step), (N,H)
  dprev_h = np.zeros((N, H))
  dprev_c = np.zeros((N, H))
  # loop backward in time
  for t in reversed(range(T)):
    # Eqn 5.22: total grad on h_t = loss grad at t + grad from step t+1, (N,H)
    dnext_h = dh[:, t, :] + dprev_h
    # Eqn 5.23: cell state only gets grad from step t+1 (not an output), (N,H)
    dnext_c = dprev_c
    # Eqn 5.24: backprop one LSTM step
    dx_t, dprev_h, dprev_c, dWx_t, dWh_t, db_t = lstm_step_backward(dnext_h, dnext_c, step_caches[t])
    # store input gradient of timestep t, (N,D)
    dx[:, t, :] = dx_t
    # Eqn 5.25: shared weights -> sum gradients over time
    dWx += dWx_t
    dWh += dWh_t
    db += db_t
  # Eqn 5.26: gradient left after t = 0 goes to h0, (N,H)
  dh0 = dprev_h
  
  return dx, dh0, dWx, dWh, db
