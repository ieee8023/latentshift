# Latent Shift - A Simple Autoencoder Approach to Counterfactual Generation


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ieee8023/latentshift/blob/main/example.ipynb)

# The idea

Read the paper: https://arxiv.org/abs/2102.09475

Watch a video: https://www.youtube.com/watch?v=1fxSDP8DheI

The main diagram:
![latentshift.gif](docs/latentshift.gif)


# Example

For a predicting of `smiling`

![gen_sequence.png](docs/gen_sequence.png)

| Counterfactual (smiling removed) | Base reconstruction |
| ----------- | ----------- |
| ![gen_bigg_cf.png](docs/gen_big_cf.png) | ![gen_bigg_base.png](docs/gen_big_base.png)  |



# Comparison

For a predicting of `pointy_nose`

![comparison.png](docs/comparison.png)

# Getting Started

```python3
# Load classifier and autoencoder
model = classifiers.FaceAttribute()
ae = autoencoders.Transformer(weights="celeba")

# Load image
input = torch.randn(1, 3, 1024, 1024)

# Defining Latent Shift module
attr = captum.attr.LatentShift(model, ae)

# Computes counterfactual for class 3.
output = attr.attribute(input, target=3)
```
