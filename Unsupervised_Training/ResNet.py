import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.layers as L

'''
Implementation of ResNet 50,101,152
with optional width parameter for 
making wider ResNets
'''

class ResNetBlock(L.Layer):
  def __init__(
      self,n_filters,regularizer=keras.regularizers.L2(1e-3),
      activation=L.ReLU,start=False,downsample=False
    ):
    super(ResNetBlock,self).__init__()
    self.regularizer = regularizer
    self.activation = activation
    self.n_filters = n_filters
    self.start = start
    self.stride = 2 if downsample else 1


    self.left_path = keras.Sequential(
        [
          L.Conv2D(self.n_filters,1,self.stride,padding='same',kernel_regularizer=self.regularizer),
          L.BatchNormalization(),
          self.activation(),
          L.Conv2D(self.n_filters,3,1,padding='same',kernel_regularizer=self.regularizer),
          L.BatchNormalization(),
          self.activation(),
          L.Conv2D(self.n_filters*4,1,1,padding='same',kernel_regularizer=self.regularizer),
          L.BatchNormalization()
        ]
    )
  
    # If first block add convolution to residual path
    # so that the number of output channels match
    if self.start:
      self.right_path = keras.Sequential(
          [
           L.Conv2D(self.n_filters*4,1,self.stride,'same',kernel_regularizer=self.regularizer),
           L.BatchNormalization()
          ]
      )

    

  def call(self,input_tensor,training=False):
    x = self.left_path(input_tensor)
    if self.start:
      y = self.right_path(input_tensor)
    else:
      y = input_tensor

    z = tf.add(x,y)
    return self.activation()(z)
      

class ResNetStack(L.Layer):
  def __init__(self,
      n_blocks,n_filters,
      first_block=False,**kwargs
    ):
    super(ResNetStack,self).__init__()
    blocks = [ 
      ResNetBlock(n_filters,start=(not i),downsample=(not i and not first_block),**kwargs) 
      for i in range(n_blocks)
    ]
    self.stack = keras.Sequential(
        blocks
    )
    self.out_dim = n_filters * 4

  def call(self,input_tensor,training=False):
    return self.stack(input_tensor) 

def ResNet(n_layers,width=1,input_shape=None,**kwargs):
    if n_layers not in [50,101,152]:
      raise ValueError
    
    model = keras.Sequential([
        L.Input(input_shape),
        L.Conv2D(64,7,2,padding='same'),
        L.BatchNormalization(),
        L.ReLU(),
        L.MaxPool2D(3,2,'same')
        
    ],name=f'ResNet{n_layers}')


    model_specs = {
        50: [3,4,6,3],
        101: [3,4,23,3],
        152: [3,8,36,3]
    }
    filters = list(map(lambda x: x*width,[64,128,256,512]))

    for i,(stack,f) in enumerate(zip(model_specs[n_layers],filters)):
      if i == 0:
        stack = ResNetStack(stack,f,first_block=True)
      else:
        stack = ResNetStack(stack,f,first_block=False)
      model.add(stack)
    model.add(L.GlobalAveragePooling2D())
    return model


