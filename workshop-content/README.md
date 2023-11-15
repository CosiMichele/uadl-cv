# Computer Vision: Image and Video Analysis


## Why Computer Vision (CV)?

In today's technological landscape, the implementation of Computer Vision (CV) stands as a transformative force, wielding a myriad of possibilities and advancements across diverse domains. Its applicability spans a wide spectrum, from [bolstering medical diagnostics by swiftly analyzing MRI scans](https://www.nature.com/articles/d41586-023-03482-9) to [fortifying autonomous vehicles with the vision to navigate complex terrains](https://www.nytimes.com/2023/08/21/technology/waymo-driverless-cars-san-francisco.html). 

Computer Vision has already made significant strides, revolutionizing fields like [retail through automated checkout systems](https://towardsdatascience.com/how-the-amazon-go-store-works-a-deep-dive-3fde9d9939e9) and [facial recognition for enhanced security measures](https://www.tsa.gov/news/press/factsheets/facial-recognition-technology). It holds the promise of simplifying daily life, automating tasks such as fruit recognition for sorting in agriculture or enhancing augmented reality experiences for immersive interactions. 

Yet, this dynamic field bears its challenges, as interpreting and understanding visual data under varying conditions remains intricate, while also offering accessible tools and methods for those delving into its realm. From the ease of leveraging Python libraries like [OpenCV](https://opencv.org/), known for its robust image processing capabilities, to the deep learning prowess of [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) for intricate neural network designs, aspiring graduates and computer engineering enthusiasts have an array of sophisticated tools at their disposal. 

The flexibility and accessibility of these libraries render the entry into Computer Vision both engaging and intellectually stimulating, fostering a rich ground for exploration and innovation among budding professionals.

### Challenges

In the realm of Computer Vision, an array of challenges persists, transcending both classical and CNN approaches. 

- **Variability and Invariance**: Images vary significantly in terms of lighting, object poses, backgrounds, and other environmental factors. *Example*: Using images of apples taken in various lighting conditions (bright sunlight, dim indoor light, and shadowed areas).

- **Object Recognition and Classification**: Accurately recognizing and categorizing objects within images, especially in cluttered scenes or for fine-grained recognition. *Example*: Presenting images of apples placed amidst a cluttered environment with other fruits or objects.

- **Semantic Understanding and Context**: Understanding the context and semantics of images, including distinguishing objects and their relationships. Example: Showcasing images where apples are placed in diverse contexts: some in a fruit bowl, some on a tree, and some in a kitchen setting. 

- **Data Annotation and Labeling**: Annotating and labeling large datasets for training deep learning models is particularly challenging for CNNs. This includes the need for large volumes of accurately labeled data, which is essential for training CNNs. *Example*: Providing datasets of various apple varieties, sizes, and conditions (ripe, unripe, different colors). 

- **Interpretability of AI Models**: Creating AI models that can explain their decision-making processes is more pertinent to CNNs. Ensuring interpretability, especially in critical applications like healthcare and autonomous vehicles, is an essential challenge for deep learning models. *Example*: Showcasing how a CNN model identifies apples in images and discussing the challenge of explaining why the model identified specific regions or features as apples. 

## A Classical Approach to Computer Vision

## Implementing Convolutional Neural Networks (CNN) to CV

CNNs are widely used in computer vision tasks like image classification, object detection, and segmentation. Revisiting CNN:

- CNNs are a class of deep neural networks designed for processing **grid-like data**, primarily used for image analysis.
- They consist of *layers* that automatically learn *hierarchical features* from input data, reducing the need for manual feature engineering.
- Convolutional layers use learnable filters to extract features from local regions of the input.
- Pooling layers downsample feature maps, preserving important information and reducing spatial dimensions.
- Fully connected layers at the end of the network make class predictions based on the learned features.
- They excel at capturing intricate patterns and are robust to variations like object positioning in images.
- Benefits include hierarchical feature learning, translation invariance, scalability, and state-of-the-art performance in visual tasks.
- CNNs have transformed how we approach image analysis, automating feature extraction and enabling end-to-end learning.

![CNN](https://miro.medium.com/v2/resize:fit:720/format:webp/1*kkyW7BR5FZJq4_oBTx3OPQ.png)

(image credits: [Towards Data Science](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939). Original image developed by [MathWorks](https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html))

### Explaining the Layers
#### The Convolutional Layer

The convolutional layer is a fundamental component of CNNs designed to efficiently process grid-like data, such as images. It plays a crucial role in extracting meaningful features from input data through the application of convolution operations, which use filters (kernels) to scan across the input creating feature maps from the original image.

![con_layer1](https://i.stack.imgur.com/Bxix6.png)

(image source: [StackOverflow](https://stackoverflow.com/questions/51008505/kernels-and-weights-in-convolutional-neural-networks))

![con_layer2](https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif)

(image credits: *Convolution*, [Wikipedia](https://en.wikipedia.org/wiki/Convolution))

In the figure above, a 3x3 kernel is applied to the values of the image. This is called a **convolutional operation** and the resulting output is referred as a **feature map**.

An additional layer, Rectified Linear Unit (ReLU) replaces negative pixes with zeroes.

#### The Pooling Layer

The Pooling Layer is often also called the downsampling layer, as it reduces the spatial size of the image. This helps with retaining important features and lowering the complexity of the image. Pooling can help with preventing overfitting by "summarizing a region" and overall computational efficiencty by reducing the computational requirements.

#### Flattening and Fully Connected Layers

The **Flattening layer** converts the 2D feature maps into a 1D vector. This transformation prepares the extracted features for input to the **fully connected layers**, which make global predictions based on the flattened features. Neurons in these layers are connected to all neurons from the previous layer. 

![flat](https://miro.medium.com/v2/resize:fit:720/format:webp/1*IWUxuBpqn2VuV-7Ubr01ng.png)

(image credits: *The Most Intuitive and Easiest Guide for Convolutional Neural Network*, [Towards Data Science](https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-convolutional-neural-network-3607be47480))
