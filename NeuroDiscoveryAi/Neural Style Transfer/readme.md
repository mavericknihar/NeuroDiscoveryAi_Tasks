Neural Style Transfer using VGG19

Overview

This project implements Neural Style Transfer (NST), a technique introduced by Gatys et al. that combines the content of one image with the artistic style of another image.
The main idea is to preserve the structure and objects present in the content image while transferring artistic textures, colors, and patterns from a style image.
For this implementation, I used a pre-trained VGG19 network as a fixed feature extractor and optimized the generated image using content loss and style loss.

Objective

The goal of this project was to understand:
How pre-trained convolutional neural networks can represent visual information.
How content and style can be separated using feature maps.
How Gram Matrices can be used to capture artistic style.
How optimization can be performed directly on an image rather than on network weights.

Understanding the Approach

Unlike traditional deep learning tasks where we train a model, Neural Style Transfer takes a different approach.
Instead of training VGG19, the network is kept frozen and used only for feature extraction.
The process involves three images:
1. Content Image
The image whose structure and objects we want to preserve.
Example:
Golden Gate Bridge
2. Style Image
The image whose artistic appearance we want to transfer.
Examples:
Van Gogh's Starry Night
The Great Wave off Kanagawa
3. Generated Image
The image that is gradually optimized to contain both the content and style characteristics.

Why VGG19?

VGG19 was chosen because it produces hierarchical feature representations at different depths of the network.
Early layers capture:
Edges
Colors
Simple textures
Deeper layers capture:
Shapes
Objects
Semantic structure
This makes VGG19 highly suitable for separating content and style information.

Content Representation

Content information is extracted from deeper convolutional layers.
In this implementation, the layer:
conv4_2
was used to represent content.
This layer captures high-level information such as:
Object shapes
Spatial layout
Overall scene structure
The content loss ensures that the generated image preserves these characteristics.

Style Representation

Style information is extracted from multiple convolutional layers.
The following layers were used:
conv1_1
conv2_1
conv3_1
conv4_1
conv5_1
These layers capture:
Colors
Brush strokes
Textures
Artistic patterns
To represent style, Gram Matrices were computed from the feature maps.
Gram Matrices measure relationships between different feature maps rather than their exact locations.
This allows the model to capture artistic textures independently of image structure.

Loss Functions

Content Loss
Content Loss measures how different the generated image is from the original content image.
Its purpose is to preserve:
Scene layout
Object shapes
Structural information

Style Loss

Style Loss measures how different the generated image is from the style image.
Its purpose is to transfer:
Artistic textures
Color distributions
Visual patterns

Total Loss

The final objective combines both losses:
Total Loss = Style Loss + Content Loss
The balance between style and content is controlled using weighting parameters.
A higher style weight produces stronger artistic effects.
A higher content weight preserves more of the original image.

Optimization Process

The generated image is initialized and then repeatedly updated through backpropagation.
The optimization loop follows these steps:
Pass the generated image through VGG19.
Extract content and style features.
Compute content loss.
Compute style loss.
Compute total loss.
Update the image pixels using Adam optimizer.
This process continues until the generated image achieves a balance between content preservation and style transfer.

Experimental Results

Experiment 1

Content Image

[NeuroDiscoveryAi/Neural Style Transfer/images/golden gate.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/golden%20gate.jpg)

Style Image

[NeuroDiscoveryAi/Neural Style Transfer/images/van gogh starry night](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/van%20gogh%20starry%20night)

Generated Image

[NeuroDiscoveryAi/Neural Style Transfer/Output_NST/output_styled_image3.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/Output_NST/output_styled_image3.jpg)

Observations

The generated image successfully preserved the structure of the Golden Gate Bridge while incorporating the characteristic swirling brush strokes and color palette of Van Gogh's Starry Night.
The sky region showed the strongest style transfer effect, while major structural elements such as the bridge remained recognizable.





Experiment 2

Content Image

[NeuroDiscoveryAi/Neural Style Transfer/images/golden gate.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/golden%20gate.jpg)

Style Image

[NeuroDiscoveryAi/Neural Style Transfer/images/great wave.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/great%20wave.jpg)

Generated Image

[NeuroDiscoveryAi/Neural Style Transfer/Output_NST/output_styled_image1.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/Output_NST/output_styled_image1.jpg)

Observations

The generated image preserved the bridge structure while introducing wave-like textures and artistic patterns inspired by The Great Wave.
Compared to the Van Gogh experiment, this output exhibited stronger texture transfer and more pronounced artistic edges.




Experiment 3

Content Image

[NeuroDiscoveryAi/Neural Style Transfer/images/van gogh starry night](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/van%20gogh%20starry%20night)

Style Image

[NeuroDiscoveryAi/Neural Style Transfer/images/great wave.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/images/great%20wave.jpg)

Generated Image

[NeuroDiscoveryAi/Neural Style Transfer/Output_NST/output_styled_image6.jpg](https://github.com/mavericknihar/NeuroDiscoveryAi_Tasks/blob/main/NeuroDiscoveryAi/Neural%20Style%20Transfer/Output_NST/output_styled_image6.jpg)

Observations

This experiment explored style-to-style transfer.
The output combined visual characteristics from both artworks, producing a hybrid appearance that contained swirling patterns from Starry Night and wave-inspired textures from The Great Wave.
The resulting image demonstrated how Neural Style Transfer can blend multiple artistic styles into a single composition.

Challenges Faced

During implementation, several challenges were encountered:
Balancing Style and Content
Finding the correct balance between style loss and content loss required experimentation.
Very high style weights often distorted the original content, while low style weights produced weak artistic effects.

Optimization Stability

Different optimization settings produced significantly different outputs.
Adjusting learning rates and style weights was important for achieving visually pleasing results.

Understanding Feature Representations

One of the most interesting parts of this project was understanding how different VGG19 layers capture different levels of information and how Gram Matrices encode artistic style.


Key Learnings

Through this project, I learned:
How feature maps represent visual information.
Why VGG19 is commonly used for style transfer.
How Gram Matrices capture artistic textures.
The difference between content and style representations.
How optimization can be performed directly on images rather than model parameters.

References

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style.
PyTorch Documentation.

