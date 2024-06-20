from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, LeakyReLU, BatchNormalization, \
    Activation, Add, Dropout, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations

'''
Human Columns:,

'''

def res_tf(filter, kernel_size=3, stride=1, name=None): 
    '''The identity resnet block that skips 3 layers.
    strides := the strides for dimension reduction
    filters := the 2 sizes for the convolutional layer
    '''
    def model(x):
        # Calculate shortcut
        shortcut = Conv2D(4*filter, 1, stride, name=name+'_0_conv')(x)
        shortcut = BatchNormalization(name=name+'_0_bn')(shortcut)

        #first block 
        x = Conv2D(filter, 1, stride, name=name+'_1_conv')(x)
        x = BatchNormalization(name=name+'_1_bn')(x)
        x = Activation(activations.relu, name=name+'_1_relu')(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(filter, kernel_size, padding='same', name=name+'_2_conv')(x)
        x = BatchNormalization(name=name+'_2_bn')(x)
        x = Activation(activations.relu, name=name+'_2_relu')(x)

        # third block activation used after adding the input
        x = Conv2D(4*filter, 1, name=name+'_3_conv')(x)
        x = BatchNormalization(name=name+'_3_bn')(x)

        # add the input 
        x = Add(name=name+'_add')([shortcut, x])
        x = Activation(activations.relu, name=name+'_output')(x)

        return x
    return model

def res_identity(filters, scaling=1): 
    '''The identity resnet block that skips 3 layers.
    filters := the 2 sizes for the convolutional layer
    scaling := the scaling factor for the original input
    '''
    def model(x):
        x_skip = scaling * x
        f1, f2 = filters

        #first block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # add the input 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x
    return model

def res_conv(strides, filters): 
    '''The identity resnet block that skips 3 layers.
    strides := the strides for dimension reduction
    filters := the 2 sizes for the convolutional layer
    '''
    def model(x):
        x_skip = x # this will be used for addition with the residual block 
        f1, f2 = filters

        #first block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(strides, strides), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # Update to the input feature
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(strides, strides), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add the input 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x
    return model

def res_dropout(rate, filters): 
    '''The identity resnet block that skips 3 layers.
    rate := the dropout rate of the input
    filters := the 2 sizes for the convolutional layer
    '''
    def model(x):
        x_skip = Dropout(rate)(x) # this will be used for addition with the residual block 
        f1, f2 = filters

        #first block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        #second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # add the input 
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)

        return x
    return model


# Tools for machine learning
imageSize = (224,224,1)
latentSize = 3

humanCols_joint = []
for count in range(30):
    if count not in [0, 2, 18, 3, 8, 13, 19, 24, 26]: # Removed wrist/pinky zeros, then duplicates, then undesired
        for suffix in ['x','y','z']:
            humanCols_joint.append('joint{}_{}'.format(count,suffix))

humanCols_angle = [
    'rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_FFJ4','rh_MFJ1','rh_MFJ2','rh_MFJ3','rh_MFJ4','rh_RFJ1','rh_RFJ2','rh_RFJ3','rh_RFJ4','rh_LFJ1','rh_LFJ2','rh_LFJ3','rh_LFJ4','rh_LFJ5','rh_THJ1','rh_THJ2','rh_THJ3','rh_THJ4','rh_THJ5'
]
shadow5Cols = [
    'rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_FFJ4','rh_MFJ1','rh_MFJ2','rh_MFJ3','rh_MFJ4','rh_RFJ1','rh_RFJ2','rh_RFJ3','rh_RFJ4','rh_LFJ1','rh_LFJ2','rh_LFJ3','rh_LFJ4','rh_LFJ5','rh_THJ1','rh_THJ2','rh_THJ3','rh_THJ4','rh_THJ5'
]
shadow4Cols = [
    'rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_FFJ4','rh_MFJ1','rh_MFJ2','rh_MFJ3','rh_MFJ4','rh_RFJ1','rh_RFJ2','rh_RFJ3','rh_RFJ4','rh_THJ1','rh_THJ2','rh_THJ4','rh_THJ5'
]
shadow3Cols = [
    'rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_FFJ4','rh_RFJ1','rh_RFJ2','rh_RFJ3','rh_RFJ4','rh_THJ1','rh_THJ2','rh_THJ4','rh_THJ5'
]
allegroCols = [
    'rh_FFJ1','rh_FFJ2','rh_FFJ3','rh_FFJ4','rh_MFJ1','rh_MFJ2','rh_MFJ3','rh_MFJ4','rh_PFJ1','rh_PFJ2','rh_PFJ3','rh_PFJ4','rh_THJ1','rh_THJ2','rh_THJ3','rh_THJ4'
]
jointCounts = {
    'Human_Joint': len(humanCols_joint),
    'Human_Angle': len(humanCols_angle),
    'Shadow5': len(shadow5Cols),
    'Shadow4': len(shadow4Cols),
    'Shadow3': len(shadow3Cols)
}

class EncoderModel(Model):
    def __init__(self, *args, **kwargs):
        super(EncoderModel, self).__init__(*args, **kwargs)
        # Determine Encoder Layer
        #VGG16, ResNet50
        self.stack = Concatenate()
        self.base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')

        #encodeLayers.append(tempBase)
        
        #self.resconv1 = res_conv(3, (1, 64))
        #self.resconv2 = res_conv(3, (64, 32))
        #self.resconv3 = res_conv(3, (32, 32))
        

        self.pooling = GlobalAveragePooling2D()
        self.encode1 = Dense(1000, LeakyReLU())
        self.encode2 = Dense(500, LeakyReLU())
        self.encode3 = Dropout(.3)
        self.encode4 = Dense(250, LeakyReLU())
        self.encode5 = Dropout(.3)
        self.encode6 = Dense(100, LeakyReLU())
        self.encode7 = Dropout(.2)
        self.encode8 = Dense(20, LeakyReLU())
        self.encode9 = Dense(3, activations.sigmoid)
    
        for layer in self.base.layers:
            layer.trainable = False

    def call(self, input, training=False):
        x = self.stack([input] * 3)
        x = self.base(x)
        #x = self.resconv1(input)
        #x = self.resconv2(x)
        #x = self.resconv3(x)

        #x = self.pooling(x)
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x, training=training)
        x = self.encode4(x)
        x = self.encode5(x, training=training)
        x = self.encode6(x)
        x = self.encode7(x, training=training)
        x = self.encode8(x)
        x = self.encode9(x)

        return x


class AngleDecoderModel(Model):
    def __init__(self, output_size, *args, **kwargs):
        super(AngleDecoderModel, self).__init__(*args, **kwargs)
        self.decode1 = Dense(5, LeakyReLU())
        self.decode2 = Dense(output_size, activations.sigmoid)

    def call(self, input, training=False):
        x = self.decode1(input)
        x = self.decode2(x)
        return x


def createHumanJointDecoder():
    decoderLayers = [
        Dense(5, activations.sigmoid),
        Dense(10, activations.sigmoid),
        Dense(20, activations.sigmoid),
        Dense(30, activations.sigmoid),
        Dense(40, activations.sigmoid),
        Dense(len(humanCols_joint))
    ]
    return decoderLayers

def createHumanAngleDecoder():
    decoderLayers = [
        Dense(5, LeakyReLU()),
        #Dense(64, LeakyReLU()),
        #Dense(32, LeakyReLU()),
        Dense(len(humanCols_angle), activations.sigmoid)
    ]
    print(len(humanCols_angle))
    return decoderLayers

def createShadow5Decoder():
    decoderLayers = [
        Dense(5, LeakyReLU()),
        #Dense(64, LeakyReLU()),
        #Dense(32, LeakyReLU()),
        Dense(len(shadow5Cols),"sigmoid")
    ]
    return decoderLayers

def createShadow4Decoder():
    decoderLayers = [
        Dense(5, LeakyReLU()),
        #Dense(64, LeakyReLU()),
        #Dense(32, LeakyReLU()),
        Dense(len(shadow4Cols),"sigmoid")
    ]
    return decoderLayers

def createShadow3Decoder():
    decoderLayers = [
        Dense(5, LeakyReLU()),
        #Dense(64, LeakyReLU()),
        #Dense(32, LeakyReLU()),
        Dense(len(shadow3Cols),"sigmoid")
    ]
    return decoderLayers

def createAllegroDecoder():
    decoderLayers = [
        Dense(5, LeakyReLU()),
        #Dense(64, LeakyReLU()),
        #Dense(32, LeakyReLU()),
        Dense(len(allegroCols),"sigmoid")
    ]
    return decoderLayers
