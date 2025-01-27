from tensorflow.keras.layers import Dense, LeakyReLU, \
    Dropout, GlobalAveragePooling2D, Concatenate, Conv2d, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations


# Tools for machine learning
imageSize = (224,224,1)

humanCols_joint = []
for count in range(30):
    if count not in [0, 2, 18, 3, 8, 13, 19, 24, 26]: # Removed wrist/pinky zeros, then duplicates, then undesired
        for suffix in ('x','y','z'):
            humanCols_joint.append(f'joint{count}_{suffix}')

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
    def __init__(self, latentSize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine Encoder Layer
        #VGG16, ResNet50
        #self.stack = Concatenate()
        #self.base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')

        #encodeLayers.append(tempBase)
        
        #self.resconv1 = res_conv(3, (1, 64))
        #self.resconv2 = res_conv(3, (64, 32))
        #self.resconv3 = res_conv(3, (32, 32))

        self.conv1 = Conv2d(64, (5, 5), strides=3, activation='relu', input_shape=imageSize)
        self.max1 = MaxPooling2D((3,3))
        self.conv2 = Conv2d(64, (5, 5), strides=3, activation='relu')
        self.max2 = MaxPooling2D((3,3))
        self.conv2 = Conv2d(64, (3, 3), strides=2, activation='relu')
        self.max2 = MaxPooling2D((2,2))
        self.conv3 = Conv2d(64, (3, 3), strides=2, activation='relu')

        self.pooling = GlobalAveragePooling2D()
        self.encode1 = Dense(1000, LeakyReLU())
        self.encode2 = Dense(500, LeakyReLU())
        self.encode3 = Dropout(.3)
        self.encode4 = Dense(250, LeakyReLU())
        self.encode5 = Dropout(.3)
        self.encode6 = Dense(100, LeakyReLU())
        self.encode7 = Dropout(.2)
        self.encode8 = Dense(20, LeakyReLU())
        self.encode9 = Dense(latentSize, activations.sigmoid)
    
        #for layer in self.base.layers:
        #    layer.trainable = False

    def call(self, inputs, training=False):
        #x = self.stack([inputs] * 3)
        #x = self.base(x)
        
        #x = self.resconv1(input)
        #x = self.resconv2(x)
        #x = self.resconv3(x)
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)

        x = self.pooling(x)
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
        super().__init__(*args, **kwargs)
        self.decode1 = Dense(5, LeakyReLU())
        self.decode2 = Dense(output_size, activations.sigmoid)

    def call(self, inputs, training=False):
        x = self.decode1(inputs)
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
