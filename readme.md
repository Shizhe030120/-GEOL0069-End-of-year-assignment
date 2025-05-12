# SatVision: Advanced Land Cover and Land Use Classification Using Sentinel-2 Imagery

## Project Overview
SatVision focuses on land cover and land use classification utilizing Sentinel-2 satellite imagery combined with deep learning techniques. Through an innovative hybrid model architecture that integrates the local feature extraction capabilities of Convolutional Neural Networks with the global feature fusion capabilities of Transformers, we've achieved highly accurate classification results that outperform traditional models such as AlexNet, ResNet50, and pure Transformer approaches.

## Problem Background

### Why Land Cover and Land Use Classification?
Land cover and land use classification is crucial for:
- **Environmental Monitoring**: Tracking deforestation, urban expansion, and ecosystem changes
- **Resource Management**: Optimizing agricultural land, water resources, and natural resource utilization
- **Urban Planning**: Supporting sustainable development and land use decisions
- **Disaster Management**: Assessing flood, drought, and fire risks and impacts
- **Climate Change Research**: Monitoring land use changes affecting carbon emissions and climate

### Challenges
Using satellite imagery for land cover and land use classification faces multiple challenges:
- **Data Complexity**: Sentinel-2 images contain multiple spectral bands requiring effective processing and integration
- **Spatial Context**: Focusing solely on local features may lead to classification errors, necessitating consideration of global spatial relationships
- **Seasonal Variations**: The same area may display different characteristics across seasons
- **Classification Accuracy**: Traditional methods perform poorly on complex terrain and mixed land use types
- **Computational Efficiency**: Need to optimize computational resource usage while maintaining high accuracy

## Innovative Solution

We present the UltraLightNet-Transformer hybrid model, effectively combining the strengths of convolutional neural networks and Transformers.

### Model Architecture

Our hybrid model (CombinedNet) consists of three core components:

1. **UltraLightNet**: A lightweight CNN architecture using depth-separable convolutions and efficient parameter design, specialized for extracting local spatial features
   - Uses grouped convolutions to reduce parameter count
   - Employs SiLU activation functions to enhance non-linear expressiveness
   - Strategic pooling to preserve key feature information

2. **Attention Module**: Multi-head self-attention mechanism capturing global dependencies and long-range feature interactions within images
   - Implements pixel-level global context awareness
   - Enhances understanding of complex land cover patterns

3. **Classifier**: Optimized fully connected layer design mapping extracted features to land classification categories

### Technical Innovations

- **Depth-Separable Convolutions**: Significantly reduces model parameters and computational load while maintaining feature extraction capabilities
- **Multi-Head Self-Attention Mechanism**: Learns feature relationships from different perspectives, enhancing feature representation
- **Hybrid Architecture Design**: Seamlessly integrates the local feature extraction of convolutional operations with the global context modeling of Transformers
- **Training Time Optimization**: Innovative model architecture design substantially shortens the training cycle, 62.2% faster than pure Transformer models
- **Lightweight Efficiency**: Our model offers higher computational efficiency with superior performance compared to pure Transformer or large CNN models

## Comparative Experiments

We compared our hybrid model with the following benchmark models:
- AlexNet: Classic CNN architecture
- ResNet50: Modern residual network architecture
- Transformer: Pure attention mechanism architecture

Our model outperformed all benchmarks across multiple evaluation metrics:

| Model | Accuracy (%) | Training Time (s) |
|-------|--------------|-------------------|
| AlexNet | 86% | 212s |
| ResNet50 | 90% | 820s |
| Transformer | 82% | 1862s |
| UltraLightNet-Transformer | 92% | 363s |

### Training Efficiency Advantages

Our model not only surpasses all benchmark models in accuracy but remarkably reduces training time costs while maintaining optimal performance:
- 55.7% reduction in training time compared to ResNet50, with a 2% increase in accuracy
- 80.5% reduction in training time compared to Transformer, with a 10% increase in accuracy
- Slightly longer training time than AlexNet, but with a 6% increase in accuracy

This efficiency stems from our model architecture innovations:
- Lightweight convolutional structure reduces parameter count and computational complexity
- Depth-separable convolutions significantly lower computational overhead
- Effective integration of attention mechanisms avoids redundant calculations
- Optimized model structure enables more efficient gradient propagation

## Application Value

Our model can be widely applied to:
- Agricultural monitoring and planning
- Urban expansion analysis
- Environmental protection and ecological restoration
- Land use policy formulation
- Climate change impact assessment

## Installation and Usage

### Requirements
```
Python 3.8+
PyTorch 1.9+
torchvision
numpy
matplotlib
scikit-learn
```

### Model Training
```python
# Model training example code
from model import CombinedNet
import torch

# Initialize model
model = CombinedNet(num_classes=10)  # Adjust the number of classes according to your classification task

# Prepare data loaders
# ...

# Train model
# ...
```

Our training results show the model's performance across different epochs:

![训练精确度和损失曲线](images/metrics_curve.png)
![验证集上的分类指标](images/classification_report.png)
![混淆矩阵](images/confusion_matrix.png)

### Inference Usage
```python
# Model inference example code
import torch
from model import CombinedNet

# Load pre-trained model
model = CombinedNet(num_classes=10)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Inference process
# ...
```

Visual results of our model's inference on real-world scenarios:

![可视化1](images/River_1_prediction.png)
![可视化2](images/Forest_2_prediction.png)
![可视化3](images/AnnualCrop_3_prediction.png)

## Conclusion

Our proposed UltraLightNet-Transformer hybrid model successfully improves the accuracy of land cover and land use classification for Sentinel-2 satellite imagery by combining the local feature extraction capabilities of convolutional neural networks with the global feature fusion capabilities of Transformers. Experimental results show that our model achieves 92% accuracy, significantly outperforming traditional AlexNet (86%), ResNet50 (90%), and pure Transformer models (82%), while substantially reducing training time, achieving both performance and efficiency breakthroughs.

Notably, our model achieves the highest accuracy while reducing training time by 80.5% compared to current popular pure Transformer models (1862s vs 363s) and by 55.7% compared to ResNet50 (820s vs 363s). This significant improvement in training efficiency enables faster model deployment in practical applications, greatly reducing computational resource consumption and research costs.

This innovative approach provides new insights for remote sensing image analysis and land cover classification, with promising applications in environmental monitoring, resource management, urban planning, and other fields, particularly suitable for scenarios requiring rapid training and deployment.

## Future Work

- Explore additional spectral band fusion methods
- Enhance temporal data analysis capabilities
- Optimize model inference speed for real-time analysis
- Expand to more application scenarios, such as precision agriculture and natural disaster monitoring