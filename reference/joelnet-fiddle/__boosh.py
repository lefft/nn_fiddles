# question: 
#   - in `train(net, ...`, does call `net.backward(grad)` 
#     change state of `net` globally? 
# answer: 
#   - yes. run `[*net.params_and_grads()]` before and 
#     after calling `train(net)` to see 


#################################################################
### EXAMPLE 1: LOGICAL XOR --------------------------------------
### xor example worked thru w annotations + edits 
import inspect

import numpy as np

from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh

from joelnet.layers import Sigmoid # tim def'ned, sept16 

# logical xor is defined as: 
#   xor(bool1,bool2) := false if bool1==bool2 else true 
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])


# OR: USE SIGMOID ACTIVATION LAYER I IMPLEMENTED 
# NOTE: NOT QUITE AS ACCURATE, BUT PRETTY CLOSE...
# Sigmoid() # instead of Tanh()

# instantiate the net (supply layers as iterable, here a list) 
net = NeuralNet([
  # layer 1: input, takes TODO
  Linear(input_size=2, output_size=2),
  # layer 2: activation layer, hyperbolic tangent 
  Tanh(),
  # layer 3: output, returns TODO
  Linear(input_size=2, output_size=2)])

# check out attrs of the net + its class 
#   inspect.signature(net.backward)
#   vars(NeuralNet)
# 
# NeuralNet methods:
#   - .forward(inputs): propogate inputs to next layer 
#   - .backward(grad): propogate gradients to previous layer 
#   - .params_and_grads(): generator yielding params and gradients
train(net, inputs, targets)

for x, y in zip(inputs, targets):
  predicted = net.forward(x)
  # WHY ROUND NO WORKIE RITE?!?! (w round or np.round, same...)
  # predicted = [np.round(pred, 4) for pred in predicted]
  print(x, predicted, y)

# predicted values over the input grid: 
#   input   prediction (in [-1,1])    output 
#   [0 0] [9.99999e-01  -1.65689e-09] [1 0]
#   [1 0] [1.38095e-09   1.00000e+00] [0 1]
#   [0 1] [1.44946e-09   1.00000e+00] [0 1]
#   [1 1] [9.99999e-01  -1.97898e-09] [1 0]



#################################################################
### EXAMPLE 2: FIZZBUZZ -----------------------------------------
### fizzbuzz example worked thru w annotations + edits 
from typing import List

import numpy as np

from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh
from joelnet.optim import SGD

# fizzbuzz is defined as: 
#   fizzbuzz(x) := 
#       (i) 'fizzbuzz' if 3 and 5 divide x;
#      (ii) 'fizz'     if 3 but not 5 divides x;
#     (iii) 'buzz'     if 5 but not 3 divides x; 
#      (iv) 'x'        otherwise
# 
# in implementation below, label indices are like this: 
#   [iv (else: x), ii (3: fizz), iii (5: buzz), i (3+5: fizzbuzz)]
def fizz_buzz_encode(x: int) -> List[int]:
  if x % 15 == 0:
    return [0, 0, 0, 1]
  elif x % 5 == 0:
    return [0, 0, 1, 0]
  elif x % 3 == 0:
    return [0, 1, 0, 0]
  else:
    return [1, 0, 0, 0]


# NOTE: for bitwise op `>>`, see:
#         - https://wiki.python.org/moin/BitwiseOperators
# QUESTION: does 10 digits limit vals of x to 0:99?? 
def binary_encode(x: int) -> List[int]:
  # 10 digit binary encoding of x
  return [x >> i & 1 for i in range(10)]


# input + output data (ints 100:1023) 
inputs = np.array([binary_encode(x) for x in range(101, 1024)])
targets = np.array([fizz_buzz_encode(x) for x in range(101, 1024)])


# define the network 
net = NeuralNet([Linear(input_size=10, output_size=50),
                 Tanh(),
                 Linear(input_size=50, output_size=4)])

# train the network 
train(net, inputs, targets,
      num_epochs=5000, optimizer=SGD(lr=0.001))

# generate predictions on 1:100 + print them out 
for x in range(1, 101):
  predicted = net.forward(binary_encode(x))
  predicted_idx = np.argmax(predicted)
  actual_idx = np.argmax(fizz_buzz_encode(x))
  labels = [str(x), "fizz", "buzz", "fizzbuzz"]
  print(x, labels[predicted_idx], labels[actual_idx])





### SCRATCH AREYAYA ------------------------------------------ 


### mess araunde w binary encoding 
def binary_encode(x: int) -> List[int]:
  # 10 digit binary encoding of x
  return [x >> i & 1 for i in range(10)]

# or: [binary_encode(x) for x in range(11)]
binary0to10 = [*map(binary_encode, range(11))]

binary0to10 = [[0,0,0,0,0,0,0,0,0,0]] + binary0to10 # make nonunique

# doesnt work bc list not hashable: `set(binary0to10)`
len(set(tuple(l) for l in binary0to10))
len(binary0to10)
[list(l) for l in set(tuple(l) for l in binary0to10)]

# make nonunique + use unique2() to operate on lists 
# binary0to10 = [[0,0,0,0,0,0,0,0,0,0]] + binary0to10 
# [unique2(binary0to10)[x] == binary0to10[x+1] for x in range(10)]


from typing import Union

# want to require that list elems have same type as input... 
# but realizing this is not possible bc no data struct like atomic vecs...
# want smthg that does: `fun(l:list) -> List[type(l's elements)]`
def unique3(l:list) -> List[int]: # Union[int, float]]:
  out = []
  # out = eval('type(l)')() # TODO: allow e.g. tuples too 
  for elem in l: 
    if not elem in out: out.append(elem)
  return out

def unique2(l):
  out = []
  for elem in l: 
    if not elem in out: out.append(elem)
  return out


unique2([1,2,3,1,4])
unique2((1,2,3,1,4))


# nice alaising ex from `docs.python.org/3/library/typing.html`
Vector = List[float]
def scale(scalar: float, vector: Vector) -> Vector:
  return [scalar * num for num in vector]

scale(2.0, [1.0, -4.2, 5.4])
# but then there is some weird shyte e.g.  
scale(str(5), [1, 2, 3]) # why return bc not list of floats?!?! 




#################################################################
### EXAMPLE 3: make one up!!! -----------------------------
### [TODO: make up a reasonable clf problem w some fake data] 
import numpy as np

from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh 
from joelnet.optim import SGD

# TODO: 
#   x- fit some regression models on xor + fizzbuzz 
#   - implement Sigmoid activation layer 
#   - do xor + fizzbuzz w sigmoid instead of tanh
#   - another activation func to tell diff 
#   - either write preds to plot in r or just plot them 
#   - identify a reasonable problem to model 
#   - fit sklearn models + write down performance 
#   - 


