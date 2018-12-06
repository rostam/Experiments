import tensorflow as tf
import tensorflow

tfe = tf.contrib.eager

def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

square(3.)  # => 9.0
grad(3.)    # => [6.0]

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]

# With flow control:
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)
# print(a)

# grad(3.)   # => [1.0]
# grad(-3.)  # => [-1.0]
