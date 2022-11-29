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

## Animations/GIFs

| Smiling | Arched Eyebrows| Mouth Slightly Open | Young |
| ----------- | ----------- | ------ |------ |
| <video src="https://user-images.githubusercontent.com/446367/204653789-2838eefb-fe03-4f3e-94d0-0f7990a7aca4.mp4"> |  <video src="https://user-images.githubusercontent.com/446367/204653885-c4902c93-02bc-45ba-8d18-df07fca7e9da.mp4"> | <video src="https://user-images.githubusercontent.com/446367/204653897-1c2e18a4-6a5d-4495-b565-0ea884402063.mp4"> | <video src="https://user-images.githubusercontent.com/446367/204653911-3e15b6bb-3dc1-4102-a2a3-52ad5f6608a5.mp4"> | 






















  
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
