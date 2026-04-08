# AMLOD: Diag-OCP Optimizer Plugin

A universal PyTorch optimizer plugin based on the **Diagonal Optimal Conjugate Projection (Diag-OCP)** framework.

## Features

- **Second-Order Optimization**: Efficiently estimates the Hessian diagonal using the Hutchinson method.
- **Adaptive Step Sizes**: Dynamically scales learning rates based on local curvature for faster convergence.
- **Project-Independent**: Zero external dependencies (only requires `torch`).
- **Standard API**: Compatible with standard PyTorch training loops (`zero_grad` -> `backward` -> `step`).

## Installation

Simply copy `diag_ocp.py` to your project directory.

## Usage

```python
import torch
from diag_ocp import DiagOCP

model = YourModel()
optimizer = DiagOCP(model.parameters(), lr=1e-3, update_each=4)

# Training loop
for input, target in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(input), target)
    
    # Required: retain graph for Hessian calculation
    loss.backward(create_graph=True)
    
    optimizer.step()
```

## Citation

If you use this optimizer in your research, please cite the original AMLOD project/paper.
