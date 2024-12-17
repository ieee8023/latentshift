
import torch
import torch.nn as nn
import torch.optim as optim

class Wachter():
    def __init__(self, model):
        """
        Initialize the Wachter counterfactual generator.

        Parameters:
        model: A PyTorch model with a forward method that takes input x and returns predictions.
        """
        self.model = model

    def attribute(self, x, target=0, target_output = 0, _lambda=1, optimizer="adam", lr=0.005, max_iter=100):
        """
        Generate counterfactual explanations for the input x.

        Parameters:
        x (torch.Tensor): The input instance for which counterfactuals are to be generated (1D tensor).
        target (int): Index of the output to reduce
        _lambda (float): Regularization strength for proximity term.
        optimizer (str): The optimizer to use ("adam" or "sgd").
        lr (float): Learning rate for the optimizer.
        max_iter (int): Maximum number of optimization iterations.

        Returns:
            attributions, a list of dicts containing the
                follow keys:
                generated_images: A list of images generated at each step along
                    the dydz vector from the smallest lambda to the largest. By
                    default the smallest lambda represents the counterfactual
                    image and the largest lambda is 0 (representing no change).
                preds: A list of the predictions of the model for each generated
                    image.
                heatmap: A heatmap indicating the pixels which change in the
                    video sequence of images.
        """
        # Ensure the input is a tensor and requires gradients
        x_cf = x.clone().detach().requires_grad_(True)
        
        # Choose optimizer
        if optimizer.lower() == "adam":
            opt = optim.Adam([x_cf], lr=lr)
        elif optimizer.lower() == "sgd":
            opt = optim.SGD([x_cf], lr=lr)
        else:
            raise ValueError("Unsupported optimizer. Use 'adam' or 'sgd'.")

        # Loss function to optimize: prediction loss + regularization (L2 proximity)
        def loss_fn(x_cf, target, original_x):
            # Prediction loss
            pred = self.model(x_cf.unsqueeze(0))  # Add batch dimension
            pred = pred[:,target]
            prediction_loss = (pred - target_output).pow(2).mean()

            # Proximity loss (L2 distance between x_cf and original x)
            proximity_loss = _lambda * torch.norm(x_cf - original_x, p=2)

            return pred.detach(), prediction_loss + proximity_loss

        # Optimization loop
        for i in range(max_iter):
            opt.zero_grad()

            # Compute loss
            pred, loss = loss_fn(x_cf, target, x)

            # Backpropagate
            loss.backward()

            # Update counterfactual instance
            opt.step()

            # Optional: Print loss for debugging
            if i % 10 == 0:
                print(f"Iteration {i}/{max_iter}, Loss: {loss.item()}, Pred {pred.item()}")
        print(f"Done, Loss: {loss.item()}, Pred {pred}")

        opred = self.model(x.unsqueeze(0)).detach().cpu().numpy()
        heatmap = torch.abs(x_cf.detach() - x.detach())
        x = x.detach().cpu()
        x_cf = x_cf.detach().cpu()
        
        return {
            "generated_images": [x_cf.numpy(), x.numpy()],
            "preds": [pred.cpu().numpy(), opred],
            "heatmap": heatmap.detach().cpu().numpy()
        }
        
        return 

# Example usage
if __name__ == "__main__":
    # Define a simple PyTorch model for demonstration
    class SimpleModel(nn.Module):
        def forward(self, x):
            return torch.sigmoid(torch.sum(x))

    # Initialize the model
    model = SimpleModel()

    # Input instance (e.g., 3 features)
    x = torch.tensor([0.5, 0.3, 0.2], requires_grad=False)

    # Initialize Wachter counterfactual generator
    wachter = Wachter(model)

    # Generate counterfactuals
    cf = wachter.generate_counterfactual(x, target=0, _lambda=1, optimizer="adam", lr=0.1, max_iter=200)

    print("Original input:", x)
    print("Counterfactual:", cf)
