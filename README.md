<script src="jquery-3.3.1.min.js"></script>
<script>
    $(function(){
        $("#softmax").load("./softmax.html");
    });
</script>

<script>
    $(function(){
        $("#selu").load("./selu.html");
    });
</script>

# Convolutional Neural Network - CIFAR 10

Authors: 

- **Jahir Gilberth Medina Lopez**
    * *USP #* **10659682**
- **Jeffry Erwin Murrugarra Llerena**
    * *USP #* **10655837** 

[**Note**](./note.html)

## Architecture

The architecture below is the final step on a very long test of architectures modifications and training variables changes **(See *Old Architectures* Section)**.

### Description

| Id Lyr.  | Lyr. Size  | Fltr. Size   | Activation    | Dropout   | Max-Pool   |    Layer Type     |
|:--------: |:------------: |:-----------:  |:----------:   |:-------:  |:-----------:  |:----------------: |
|     0     |    32x32x3    |      -        |      -        |    -      |      -        |    Input Layer    |
|     1     |      36       |     3x3       |    ReLU       |   0.0     |      -        | Convolutional  |
|     2     |      36       |     2x2       |    SeLU       |   0.2     |      -        | Convolutional  |
|     3     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|     4     |      48       |     2x2       |    SeLU       |   0.0     |      -        | Convolutional  |
|     5     |      48       |     4x4       |    ReLU       |   0.3     |      -        | Convolutional  |
|     6     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|     7     |      64       |     3x3       |    ReLU       |   0.0     |      -        | Convolutional  |
|     8     |      64       |     3x3       |    SeLU       |   0.3     |      -        | Convolutional  |
|     9     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|    10     |      500      |      -        |    SeLU       |   0.8     |      -        |  Fully Conn.  |
|    11     |      300      |      -        |    ReLU       |   0.4     |      -        |  Fully Conn.  |
|    12     |      10       |      -        |   SoftMax     |   0.0     |      -        |   Output Layer    |

> CNN Optmizer : **Ada Delta**
> 
> Loss Function : **Categorical Crossentropy**
> 
> Metrics : **Accuracy Default by Loss Function : Categorical**
> 
> Epochs : **150**
> 
> Batch Size : **1000**

### Activations Functions Formulas

* **SeLU**
    SeLU stands for ***scaled exponential linear units*** : 
    <div id="selu"></div>
    where for standard scaled inputs (mean 0, standard deviation 1), the values are α=1.6732~, λ=1.0507~. This function is monotonic, with a slope very silimar to the ReLU , but also with a slow change on it.

* **SoftMax**
    The softmax function is used in various multiclass classification methods. 
    <div id="softmax"></div>
    The output of the softmax function can be used to represent a categorical distribution.


### Architectures Details

- The use of an equal number of neurons in consecutive layers who resides before max-pooling layers aims to avoid characteristics loss or "unexpected" filter re-sizing on them, a different number of neurons per layer "improve" flexibility but also adds instability.

- Using the same activation function on layers immediately near to a max-pooling layer aids to conserve momentum in the characteristics extraction process.

- The use of a \[500-300-10\] fully connected MLP was designed desiring a slowly reduce the number of dimensions until the output layer is reached, perhaps 500 it is a lot of input neurons, but that is why we assign to that layer 0.8 of dropout probability.

## Data

At the first try, we try to load the data from a local zip file.
But this *"binaries ready dataset for python"* has errors when it's tried to be visualized, meaning the data have errors.

Instead, the online data downloaded from the Keras online Repository doesn't present those errors, being suitable to this experiment

### Batched Data (Binaries for Python)

![](./media/disk-bird.png)

### Keras CIFAR 10

![](./media/online-bird.png)

## Training

Using 150 epochs and a 1000 cases per batch, this CNN trains like was expected, converging to an *"stable"* accuracy value of 0.78-0.81~.

### Training Evolution Plot

#### First Training

![](./media/v0/accu-train.png)

![](./media/v0/loss-train.png)

#### 5 Epoch Post Training

More even, to check the training process has already archived the *"stable minimum"* for the architecture choose, we re-run the training process from was stopped.
It can see, the variation in the values of accuracy and loss are very small.

![](./media/v0/accu-post-train0.png)

![](./media/v0/loss-post-train0.png)

#### 50 Epoch Post Training

Only to have a more strong confirmation in the post-train testing, we add 50 epochs of training.

##### Accuracy

![](./media/v0/accu-post-train1.png)

![](./media/v0/diff-accu-post-train1.png)

##### Losses

![](./media/v0/loss-post-train1.png)

![](./media/v0/diff-loss-post-train1.png)

## Conclusions

- Increase the size of the data using some artificial data size augmentation
- Create 2 or more parallel CNN at the beginning, searching a better filter-size and filter selection.
- Use other family or type the optimizer function, **Ada Delta** works well, but could be improved using some ***automatic resizer gradient step function***.
- Not use **RMS prop.** as optimizer function

## Old Architectures

### v1

This code was obtained [online](https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/), and used as a dummy for learning (How to properly use Keras), more even, this kind of architecture uses a ***RMS Prop*** optimization, how works better with *Recurrent Neural Networks* and *insufficient* batch size.

#### Design

| Id Lyr.  | Lyr. Size  | Fltr. Size   | Activation    | Dropout   | Max-Pool   |    Layer Type     |
|:--------: |:------------: |:-----------:  |:----------:   |:-------:  |:-----------:  |:----------------: |
|     0     |    32x32x3    |      -        |      -        |    -      |      -        |    Input Layer    |
|     1     |      32       |     3x3       |    ReLU       |   0.0     |      -        | Convolutional  |
|     2     |      32       |     3x3       |    ReLU       |   0.25    |      -        | Convolutional  |
|     3     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|     4     |      64       |     3x3       |    ReLU       |   0.0     |      -        | Convolutional  |
|     5     |      64       |     3x3       |    ReLU       |   0.25    |      -        | Convolutional  |
|     6     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|     7     |      64       |     3x3       |    ReLU       |   0.0     |      -        | Convolutional  |
|     8     |      64       |     3x3       |    ReLU       |   0.25    |      -        | Convolutional  |
|     9     |       -       |      -        |      -        |    -      |     2x2       |      Pooling      |
|    10     |      512      |      -        |    SeLU       |   0.5     |      -        |  Fully Conn.  |
|    11     |      10       |      -        |   SoftMax     |   0.0     |      -        |   Output Layer    |

> CNN Optmizer : **RMS prop**
> 
> Loss Function : **Categorical Crossentropy**
> 
> Metrics : **Accuracy Default by Loss Function : Categorical**
>
> Epochs : **50**
> 
> Batch Size : **256**

#### Training Plot

![](./media/v1/accu.png)

![](./media/v1/loss.png)

### v2

In this version, was analyzed the batch size and with higher emphasis the number of epochs per train.
The network architecture remain the same.

#### Design

> CNN Optmizer : **RMS prop**
> 
> Loss Function : **Categorical Crossentropy**
> 
> Metrics : **Accuracy Default by Loss Function : Categorical**
>
> Epochs : **100**
> 
> Batch Size : **1000**

#### Training Plot

![](./media/v2/accu.png)

![](./media/v2/loss.png)

### v3

The idea behind modifications on this version was to find the best (more fastest) architecture. To archive that, we were continuously changing some ***layer activation functions, layer size, and dropout probability***, the architecture below shows the last changed perform over parameters.

#### Design

| Id Lyr.  | Lyr. Size  | Fltr. Size   |  Activation   | Dropout   | Max-Pool   |    Layer Type     |
|:--------: |:------------: |:-----------:  |:------------: |:-------:  |:-----------:  |:----------------: |
|     0     |    32x32x3    |      -        |       -       |    -      |      -        |    Input Layer    |
|     1     |      36       |     3x3       | Hard Sigmoid  |   0.0     |      -        | Convolutional  |
|     2     |      48       |     5x5       |     ReLU      |   0.3     |      -        | Convolutional  |
|     3     |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
|     4     |      64       |     3x3       |  Exponential  |   0.0     |      -        | Convolutional  |
|     5     |      72       |     5x5       |     ReLU      |   0.25    |      -        | Convolutional  |
|     6     |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
|     7     |      64       |     3x3       |     ReLU      |   0.0     |      -        | Convolutional  |
|     8     |      64       |     3x3       |     Tanh      |   0.4     |      -        | Convolutional  |
|     9     |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
|    10     |      400      |      -        |     ReLU      |   0.8     |      -        |  Fully Conn.  |
|    11     |      200      |      -        |     ReLU      |   0.4     |      -        |  Fully Conn.  |
|    12     |      100      |      -        |     Tanh      |   0.3     |      -        |  Fully Conn.  |
|    13     |      10       |      -        |    SoftMax    |   0.0     |      -        |   Output Layer    |

> CNN Optmizer : **RMS prop**
> 
> Loss Function : **Categorical Crossentropy**
> 
> Metrics : **Accuracy Default by Loss Function : Categorical**
>
> Epochs : **100**
> 
> Batch Size : **1000**

#### Training Plot

![](./media/v3/accu.png)

![](./media/v3/loss.png)

### v4

Using the same ***activation functions at the same layers*** we try to improve the accuracy and reduce losses; to arrive at our objective, we use a more *smooth* variation in the layers size, and increasing the number of ***epochs***.

One valuable thing learned in this variant was more about control instead of network architecture design; even increasing significantly the number of epochs, the training evolution goes from good to **worse**. Like can be see in the plot below, when the minimum was archived the ***CNN*** continues trying to perform better, then, and because we were using some non linear activation functions, the accuracy goes to a very low value following the path of our architecture in his way to *" the global minimum"* but in the wrong direction. ***In few words, using 500 epochs, we force the net to goes from a maximum to minimum and later again to the maximum (the gradient takes the wrong direction)***

#### Design

| Id Lyr.  | Lyr. Size  | Fltr. Size   |  Activation   | Dropout   | Max-Pool   |    Layer Type     |
|:--:   |:------------: |:-----------:  |:------------: |:-------:  |:-----------:  |:----------------: |
|  0    |    32x32x3    |      -        |       -       |    -      |      -        |    Input Layer    |
|  1    |      36       |     3x3       | Hard Sigmoid  |   0.0     |      -        | Convolutional  |
|  2    |      48       |     5x5       |     ReLU      |   0.3     |      -        | Convolutional  |
|  3    |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
|  4    |      48       |     3x3       |  Exponential  |   0.0     |      -        | Convolutional  |
|  5    |      48       |     5x5       |     ReLU      |   0.25    |      -        | Convolutional  |
|  6    |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
|  7    |      48       |     3x3       |     ReLU      |   0.0     |      -        | Convolutional  |
|  8    |      64       |     3x3       |     Tanh      |   0.4     |      -        | Convolutional  |
|  9    |       -       |      -        |       -       |    -      |     2x2       |      Pooling      |
| 10    |      400      |      -        |     ReLU      |   0.8     |      -        |  Fully Conn.  |
| 11    |      200      |      -        |     ReLU      |   0.4     |      -        |  Fully Conn.  |
| 12    |      100      |      -        |     Tanh      |   0.3     |      -        |  Fully Conn.  |
| 13    |      10       |      -        |    SoftMax    |   0.0     |      -        |   Output Layer    |

> CNN Optmizer : **RMS prop**
> 
> Loss Function : **Categorical Crossentropy**
> 
> Metrics : **Accuracy Default by Loss Function : Categorical**
>
> Epochs : **500**
> 
> Batch Size : **1000**

#### Training Plot

![](./media/v4/accu.png)

![](./media/v4/loss.png)