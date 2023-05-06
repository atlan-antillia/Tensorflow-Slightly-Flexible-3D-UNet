# Tensorflow-Slightly-Flexible-3D-UNet (Updated: 2023/05/06)

<h2>
1 Tensorflow 3D-UNet Model
</h2>
<p>
This is a slightly flexible 3D-UNet Model Implementation by Tensorflow 2, which is literally cusomizable by a configuration file.
</p>
In order to write the extensible UNet Model, we have used the Python scripts in the following web sites.
</p>
<pre>
1. 3D-brain-segmentation
 https://github.com/mrkolarik/3D-brain-segmentation/blob/master/3D-unet.py</pre>
</pre>
<pre>
2.Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation 
 https://www.mdpi.com/2076-3417/9/3/404
</pre>

<h2>
2 Create Tensorflow3DUNet Model
</h2>
 You can customize your Tensorflow3DUNet model by using a configration file.<br>
<pre>
; model.config
[model]
image_depth    = 32
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 6
dropout_rate   = 0.05
learning_rate  = 0.001
</pre>
This Tensorflow3DUNet model is a typical classic U-Net model, and consists of two module, Encoder and Decoder. <br>
The parameters defined in this model.config will be used to create the CNN layers of Encoder and Decoder.<br>
<table width="720" >
<tr>
<td>image_depth</td>
<td>The input image depth to the first layer of Encoder</td>
</tr>
<tr>
<td>image_width and image_height</td>
<td>
The input image size to the first layer of Encoder. 
Model.
</td>
</tr>

<tr>
<td>num_classes</td>
<td>The number of classes of dataset.</td>
</tr>

<tr>
<td>base_filters</td>
<td>The number of initial filters for Conv3D layer.</td>
</tr>
<tr>
<td>num_layers</td>
<td>
The number of blocks of Decoder and Encoder. 
</td>
</tr>

<tr>
<td>dropout_rate</td>
<td>The initial dropout_rate for Dropout layer </td>
</tr>

<tr>
<td>learning_rate</td>
<td>The learining_rate for Adam optimizer </td>
</tr>

</table>

<br>
You will pass the filename of this configuration file to <a href="./Tensorflow3DUNet.py">Tensorflow3DUNet</a> 
constructor to create your model 
in the following way:<br>
<pre>
  config_file = "./model.config"
  model       = Tensorflow3DUNet(config_file)
</pre>

The <b>create</b> method in the <b>TensorflowUNet</b> class is slightly simple as shown below. 
It mainly consists of two parts Encoder and Decoder, which are written by <b>for</b> loops depending
on <b>num_layers</b> defined in <b>model.config</b> file. 
<pre>
  def create(self, num_classes, image_depth, image_height, image_width, image_channels,
            base_filters = 16, num_layers = 5):
    # inputs
    print("Input image_height {} image_width {} image_channels {}".format(image_height, image_width, image_channels))
    inputs = Input((image_depth, image_height, image_width, image_channels))
    s= Lambda(lambda x: x / 255)(inputs)

    # Encoder
    dropout_rate = self.config.get(MODEL, "dropout_rate")
    enc         = []
    kernel_size = (3, 3, 3)
    pool_size   = (2, 2, 2)
  
    for i in range(num_layers):
      filters = base_filters * (2**i)
      c = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(s)
      c = Dropout(dropout_rate * i)(c)
      c = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(c)
      print("--- enc shape {}".format(c.shape))
      if i < (num_layers-1):
        p = MaxPool3D(pool_size=pool_size)(c)
        s = p
      enc.append(c)
    
    enc_len = len(enc)
    enc.reverse()

    n = 0
    c = enc[n]
    # --- Decoder
    for i in range(num_layers-1):
      f = enc_len - 2 - i
      filters = base_filters* (2**f)
      u = Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c)
      n += 1
      print("shape u {}".format(u.shape))
      print("shape enc[n] {}".format(enc[n].shape))

      u = concatenate([u, enc[n]], axis=4)
      u = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal', padding='same')(u)
      u = Dropout(dropout_rate * f)(u)
      u = Conv3D(filters, kernel_size, activation=relu, kernel_initializer='he_normal',padding='same')(u)
      c  = u

    # outouts
    outputs = Conv3D(num_classes, (1, 1, 1), activation='sigmoid')(c)

    # create Model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
</pre>

You can create Tensorflow3DUNet Model by running the following command.<br>
<pre>
>python Tensorflow3DUNet.py
</pre>
You will see the following summary of the model.<br>
<pre>
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 32, 256, 25  0           []
                                6, 3)]

 lambda (Lambda)                (None, 32, 256, 256  0           ['input_1[0][0]']
                                , 3)

 conv3d (Conv3D)                (None, 32, 256, 256  1312        ['lambda[0][0]']
                                , 16)

 dropout (Dropout)              (None, 32, 256, 256  0           ['conv3d[0][0]']
                                , 16)

 conv3d_1 (Conv3D)              (None, 32, 256, 256  6928        ['dropout[0][0]']
                                , 16)

 max_pooling3d (MaxPooling3D)   (None, 16, 128, 128  0           ['conv3d_1[0][0]']
                                , 16)

 conv3d_2 (Conv3D)              (None, 16, 128, 128  13856       ['max_pooling3d[0][0]']
                                , 32)

 dropout_1 (Dropout)            (None, 16, 128, 128  0           ['conv3d_2[0][0]']
                                , 32)

 conv3d_3 (Conv3D)              (None, 16, 128, 128  27680       ['dropout_1[0][0]']
                                , 32)

 max_pooling3d_1 (MaxPooling3D)  (None, 8, 64, 64, 3  0          ['conv3d_3[0][0]']
                                2)

 conv3d_4 (Conv3D)              (None, 8, 64, 64, 6  55360       ['max_pooling3d_1[0][0]']
                                4)

 dropout_2 (Dropout)            (None, 8, 64, 64, 6  0           ['conv3d_4[0][0]']
                                4)

 conv3d_5 (Conv3D)              (None, 8, 64, 64, 6  110656      ['dropout_2[0][0]']
                                4)

 max_pooling3d_2 (MaxPooling3D)  (None, 4, 32, 32, 6  0          ['conv3d_5[0][0]']
                                4)

 conv3d_6 (Conv3D)              (None, 4, 32, 32, 1  221312      ['max_pooling3d_2[0][0]']
                                28)

 dropout_3 (Dropout)            (None, 4, 32, 32, 1  0           ['conv3d_6[0][0]']
                                28)

 conv3d_7 (Conv3D)              (None, 4, 32, 32, 1  442496      ['dropout_3[0][0]']
                                28)

 max_pooling3d_3 (MaxPooling3D)  (None, 2, 16, 16, 1  0          ['conv3d_7[0][0]']
                                28)

 conv3d_8 (Conv3D)              (None, 2, 16, 16, 2  884992      ['max_pooling3d_3[0][0]']
                                56)

 dropout_4 (Dropout)            (None, 2, 16, 16, 2  0           ['conv3d_8[0][0]']
                                56)

 conv3d_9 (Conv3D)              (None, 2, 16, 16, 2  1769728     ['dropout_4[0][0]']
                                56)

 max_pooling3d_4 (MaxPooling3D)  (None, 1, 8, 8, 256  0          ['conv3d_9[0][0]']
                                )

 conv3d_10 (Conv3D)             (None, 1, 8, 8, 512  3539456     ['max_pooling3d_4[0][0]']
                                )

 dropout_5 (Dropout)            (None, 1, 8, 8, 512  0           ['conv3d_10[0][0]']
                                )

 conv3d_11 (Conv3D)             (None, 1, 8, 8, 512  7078400     ['dropout_5[0][0]']
                                )

 conv3d_transpose (Conv3DTransp  (None, 2, 16, 16, 2  1048832    ['conv3d_11[0][0]']
 ose)                           56)

 concatenate (Concatenate)      (None, 2, 16, 16, 5  0           ['conv3d_transpose[0][0]',
                                12)                               'conv3d_9[0][0]']

 conv3d_12 (Conv3D)             (None, 2, 16, 16, 2  3539200     ['concatenate[0][0]']
                                56)

 dropout_6 (Dropout)            (None, 2, 16, 16, 2  0           ['conv3d_12[0][0]']
                                56)

 conv3d_13 (Conv3D)             (None, 2, 16, 16, 2  1769728     ['dropout_6[0][0]']
                                56)

 conv3d_transpose_1 (Conv3DTran  (None, 4, 32, 32, 1  262272     ['conv3d_13[0][0]']
 spose)                         28)

 concatenate_1 (Concatenate)    (None, 4, 32, 32, 2  0           ['conv3d_transpose_1[0][0]',
                                56)                               'conv3d_7[0][0]']

 conv3d_14 (Conv3D)             (None, 4, 32, 32, 1  884864      ['concatenate_1[0][0]']
                                28)

 dropout_7 (Dropout)            (None, 4, 32, 32, 1  0           ['conv3d_14[0][0]']
                                28)

 conv3d_15 (Conv3D)             (None, 4, 32, 32, 1  442496      ['dropout_7[0][0]']
                                28)

 conv3d_transpose_2 (Conv3DTran  (None, 8, 64, 64, 6  65600      ['conv3d_15[0][0]']
 spose)                         4)

 concatenate_2 (Concatenate)    (None, 8, 64, 64, 1  0           ['conv3d_transpose_2[0][0]',
                                28)                               'conv3d_5[0][0]']

 conv3d_16 (Conv3D)             (None, 8, 64, 64, 6  221248      ['concatenate_2[0][0]']
                                4)

 dropout_8 (Dropout)            (None, 8, 64, 64, 6  0           ['conv3d_16[0][0]']
                                4)

 conv3d_17 (Conv3D)             (None, 8, 64, 64, 6  110656      ['dropout_8[0][0]']
                                4)

 conv3d_transpose_3 (Conv3DTran  (None, 16, 128, 128  16416      ['conv3d_17[0][0]']
 spose)                         , 32)

 concatenate_3 (Concatenate)    (None, 16, 128, 128  0           ['conv3d_transpose_3[0][0]',
                                , 64)                             'conv3d_3[0][0]']

 conv3d_18 (Conv3D)             (None, 16, 128, 128  55328       ['concatenate_3[0][0]']
                                , 32)

 dropout_9 (Dropout)            (None, 16, 128, 128  0           ['conv3d_18[0][0]']
                                , 32)

 conv3d_19 (Conv3D)             (None, 16, 128, 128  27680       ['dropout_9[0][0]']
                                , 32)

 conv3d_transpose_4 (Conv3DTran  (None, 32, 256, 256  4112       ['conv3d_19[0][0]']
 spose)                         , 16)

 concatenate_4 (Concatenate)    (None, 32, 256, 256  0           ['conv3d_transpose_4[0][0]',
                                , 32)                             'conv3d_1[0][0]']

 conv3d_20 (Conv3D)             (None, 32, 256, 256  13840       ['concatenate_4[0][0]']
                                , 16)

 dropout_10 (Dropout)           (None, 32, 256, 256  0           ['conv3d_20[0][0]']
                                , 16)

 conv3d_21 (Conv3D)             (None, 32, 256, 256  6928        ['dropout_10[0][0]']
                                , 16)

 conv3d_22 (Conv3D)             (None, 32, 256, 256  17          ['conv3d_21[0][0]']
                                , 1)

==================================================================================================
</pre>

