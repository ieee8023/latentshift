# Latent Shift - A Simple Autoencoder Approach to Counterfactual Generation


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ieee8023/latentshift/blob/main/example.ipynb)

# The idea

Read the paper: https://arxiv.org/abs/2102.09475

Watch a video: https://www.youtube.com/watch?v=1fxSDP8DheI

The main diagram:
![latentshift.gif](docs/latentshift.gif)


## Animations/GIFs

| Smiling <img width="5000" height="1">| Arched Eyebrows<img width="5000" height="1">|
| ----------- | ----------- |
| <img src="docs/smiling.gif" width="100%"> |  <img src="docs/arched_eyebrows.gif" width="100%"> | 

|Mouth Slightly Open <img width="5000" height="1"> | Young <img width="5000" height="1"> |
| ----------- | ----------- |
| <img src="docs/mouth_slightly_open.gif" width="100%"> |  <img src="docs/young.gif" width="100%"> | 

# Generating a transition sequence

For a predicting of `smiling`

![gen_sequence.png](docs/gen_sequence.png)

# Multiple different targets

<img src="docs/latent-shift-diffs.png" width="100%">




  
# Comparison to traditional methods 

For a predicting of `pointy_nose`

![comparison.png](docs/comparison.png)

# Getting Started

```bash
$pip install latentshift
````


```python3
import latentshift
# Load classifier and autoencoder
model = latentshift.classifiers.FaceAttribute(download=True)
ae = latentshift.autoencoders.VQGAN(weights="faceshq", download=True)

# Load image
input = torch.randn(1, 3, 1024, 1024)

# Defining Latent Shift module
attr = captum.attr.LatentShift(model, ae)

# Computes counterfactual for class 3.
output = attr.attribute(input, target=3)
```
