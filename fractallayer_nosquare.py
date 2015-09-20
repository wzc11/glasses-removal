import numpy as np
from pylearn2.models.mlp import max_pool
import theano.tensor as T
import theano.sandbox
import theano.gof
import theano

from theano.sandbox.neighbours import images2neibs
from layerbase import Layer, Param, VisLayer, LayerbasedDropOut
from theano.tensor.nnet import conv
INF = 1e10
b3tensor = T.TensorType(dtype = theano.config.floatX, broadcastable = [])

def dtypeX(val):
    return val + 0.0

class StacksampleFractal(Layer):
    
    def __init__(self, input, input_shape = None, feedval = 0.0):
        if isinstance(input, Layer):
            self.input = input.output
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        ##Only square image allowed
        #assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        output_cmb = max_pool(inputext, (3,3), (1,1), shapeext[2:])
        self.output_cmb = output_cmb
        #Separate output to 4 channels
        c00 = output_cmb[:,:,::2,::2]
        c01 = output_cmb[:,:,::2,1::2]
        c10 = output_cmb[:,:,1::2,::2]
        c11 = output_cmb[:,:,1::2,1::2]
        self.one_channel = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2

        #Combine, 2 conditions: even/odd
        odd2 = (input_shape[2]&1)==1
        odd3 = (input_shape[3]&1)==1
        joined = T.alloc(dtypeX(feedval), *((input_shape[0]*4,)+self.one_channel[1:4]))
        joined = T.set_subtensor(joined[0:self.one_channel[0],:,:,:], c00)
        joined = T.set_subtensor(joined[self.one_channel[0]:self.one_channel[0]*2,:,:,:-1] if odd3 else joined[self.one_channel[0]:self.one_channel[0]*2], c01)
        joined = T.set_subtensor(joined[self.one_channel[0]*2:self.one_channel[0]*3,:,:-1,:] if odd2 else joined[self.one_channel[0]*2:self.one_channel[0]*3], c10)
        if odd2:
            if odd3:
                joined = T.set_subtensor(joined[self.one_channel[0]*3:self.one_channel[0]*4,:,:-1,:-1], c11)
            else:
                joined = T.set_subtensor(joined[self.one_channel[0]*3:self.one_channel[0]*4,:,:-1,:], c11)
        else:
            if odd3:
                joined = T.set_subtensor(joined[self.one_channel[0]*3:self.one_channel[0]*4,:,:,:-1], c11)
            else:
                joined = T.set_subtensor(joined[self.one_channel[0]*3:self.one_channel[0]*4,:,:,:], c11)

        self.output = joined
        self.output_shape = input_shape[0]*4, self.one_channel[1], self.one_channel[2], self.one_channel[3]

class DestacksampleFractal(Layer):

    def __init__(self, input, stacksamplelayer, input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape
        assert isinstance(stacksamplelayer, StacksampleFractal)
        
        #Insert back by shape
        odd2 = (stacksamplelayer.input_shape[2]&1)==1
        odd3 = (stacksamplelayer.input_shape[3]&1)==1

        self.output_shape = stacksamplelayer.input_shape[0], input_shape[1], (input_shape[2]*2-1) if odd2 else (input_shape[2]*2), (input_shape[3]*2-1) if odd3 else (input_shape[3]*2)
        self.output = T.alloc(dtypeX(0.0), *self.output_shape)

        c00 = self.input[0:stacksamplelayer.one_channel[0]]
        c01 = self.input[stacksamplelayer.one_channel[0]:stacksamplelayer.one_channel[0]*2]
        c10 = self.input[stacksamplelayer.one_channel[0]*2:stacksamplelayer.one_channel[0]*3]
        c11 = self.input[stacksamplelayer.one_channel[0]*3:stacksamplelayer.one_channel[0]*4]
        

        self.output = T.set_subtensor(self.output[:,:,::2,::2], c00)
        self.output = T.set_subtensor(self.output[:,:,::2,1::2], c01[:,:,:,:-1] if odd3 else c01)
        self.output = T.set_subtensor(self.output[:,:,1::2,::2], c10[:,:,:-1,:] if odd2 else c10)
        if odd2:
            if odd3:
                self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:-1,:-1])
            else:
                self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:-1,:])
        else:
            if odd3:
                self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:,:-1])
            else:
                self.output = T.set_subtensor(self.output[:,:,1::2,1::2], c11[:,:,:,:])

class ShrinkshapeMeanFractal(Layer):

    def __init__(self,input,input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        ##Only square image allowed
        #assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+2, input_shape[3]+2
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        self.output_shape = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2
        
        self.output = images2neibs(inputext, (3,3), (2,2), 'ignore_borders').mean(axis=-1)
        self.output = self.output.reshape(self.output_shape)
        

class ShrinkshapeFractal(Layer):

    def __init__(self,input,input_shape = None):
        if isinstance(input, Layer):
            self.input = input.output
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape

        ##Only square image allowed
        #assert input_shape[2]==input_shape[3]

        #Extend one pixel at each direction
        shapeext = input_shape[0], input_shape[1], input_shape[2]+1, input_shape[3]+1
        inputext = T.alloc(dtypeX(-INF), *shapeext)

        inputext = T.set_subtensor(inputext[:,:,1:input_shape[2]+1,1:input_shape[3]+1], self.input)
        
        self.output = max_pool(inputext, (3,3), (2,2), shapeext[2:])
        self.output_shape = input_shape[0], input_shape[1], (input_shape[2]+1)/2, (input_shape[3]+1)/2
        print self.output_shape

class ExpandshapeFractal(Layer):

    def __init__(self, input, shrinksamplelayer, input_shape=None, calibrate = True):
        if isinstance(input, Layer):
            self.input = input.output
            if input_shape == None:
                input_shape = input.output_shape
        else:
            self.input = input
        self.input_shape = input_shape
         
        assert isinstance(shrinksamplelayer, (ShrinkshapeFractal, ShrinkshapeMeanFractal))

        if calibrate:
            cali=2
        else:
            cali=0

        self.output_shape = input_shape[0], input_shape[1]+cali, shrinksamplelayer.input_shape[2], shrinksamplelayer.input_shape[3]
        print self.output_shape

        output = T.alloc(dtypeX(0.0), *self.output_shape)
        
        odd2 = (shrinksamplelayer.input_shape[2]&1)==1
        odd3 = (shrinksamplelayer.input_shape[3]&1)==1
        #Expand data 4 fold
        output = T.set_subtensor(output[:,:input_shape[1],::2,::2], self.input)
        output = T.set_subtensor(output[:,:input_shape[1],::2,1::2], self.input[:,:,:,:-1] if odd3 else self.input)
        output = T.set_subtensor(output[:,:input_shape[1],1::2,::2], self.input[:,:,:-1,:] if odd2 else self.input)
        t = self.input[:,:,:-1,:] if odd2 else self.input
        t = t[:,:,:,:-1] if odd3 else t
        output = T.set_subtensor(output[:,:input_shape[1],1::2,1::2],t)
        
        #Feed calibrate data
        if calibrate:
            #HACK, strange
            dval = T.alloc(dtypeX(1.0), input_shape[0], shrinksamplelayer.input_shape[2], (shrinksamplelayer.input_shape[3])/2)
            output = T.set_subtensor(output[:,input_shape[1],:,1::2], dval)
            dval = T.alloc(dtypeX(1.0), input_shape[0], (shrinksamplelayer.input_shape[2])/2, shrinksamplelayer.input_shape[3])
            output = T.set_subtensor(output[:,input_shape[1]+1,1::2], dval)

        self.output = output

class AggregationLayer(Layer):

    def __init__(self, *layers):

        channels = 0
        for i in layers:
            assert isinstance(i, Layer)
            channels += i.output_shape[1]

        self.output_shape = layers[0].output_shape[0], channels, layers[0].output_shape[2], layers[0].output_shape[3]
        self.output = T.alloc(dtypeX(0.0), *self.output_shape)
        channels = 0
        for i in layers:
            self.output = T.set_subtensor(self.output[:,channels:channels+i.output_shape[1]], i.output)
            channels += i.output_shape[1]

class ConvKeepLayer(Layer, Param, VisLayer):

    def __init__(self, rng, input, filter_shape, image_shape = None, Nonlinear = "tanh", zeroone = False, inc=[0], dropout = False, dropoutrnd = None):

        if isinstance(input, Layer):
            self.input = input.output 
            if image_shape==None:
                image_shape = input.output_shape
        else:
            self.input = input
        print image_shape[1]
        #assert image_shape[1] == filter_shape[1]
        assert filter_shape[2]%2 == 1
        assert filter_shape[3]%2 == 1
        med = (filter_shape[2]-1)/2,(filter_shape[3]-1)/2

        fan_in = np.prod(filter_shape[1:])
        W_values = np.asarray(rng.uniform(
              low=-np.sqrt(0.5/fan_in),
              high=np.sqrt(0.5/fan_in),
              size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_%s'%inc[0])

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_%s'%inc[0])

        conv_out = conv.conv2d(self.input, self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode="full")
        #Get middle area
        conv_out = conv_out[:,:,med[0]:-med[0],med[1]:-med[1]]

        if Nonlinear:
            self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            if zeroone:
                self.output = (self.output+1) * 0.5
        else:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output_shape = (image_shape[0], filter_shape[0], image_shape[2], image_shape[3])
        
        #dropout = False
        if not (dropout is False): #Embed a layerwise dropout layer
            self.output = LayerbasedDropOut(self, dropoutrnd, dropout).output
        
        self.params = [self.W, self.b]

        inc[0] = inc[0]+1
 

if __name__=="__main__":
    import numpy as np
    import theano

    a=np.array([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]]],'f')

    #Make subsample fractal test layer
    inp = T.tensor4("inp")
    l1 = StacksampleFractal(inp, a.shape)
    l2 = DestacksampleFractal(l1, l1)
    f = theano.function([inp], [l1.output, l2.output])
    print f(a)

    l1 = ShrinkshapeFractal(inp, a.shape)
    l2 = ExpandshapeFractal(l1, l1)
    f = theano.function([inp], [l1.output, l2.output])
    print f(a)


