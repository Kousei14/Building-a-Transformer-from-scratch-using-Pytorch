This repository contains a full implementation of the **"Attention Is All You Need"** paper from scratch. The Transformer architecture is built entirely with custom code.
## Overview
- Complete implementation of the Transformer model from the original paper.
- Custom-built layers, attention mechanisms, and training pipeline.
- Trained a custom language translation model using the **Helsinki-NLP dataset** from Hugging Face.

## Dataset
The model is trained on the **Helsinki-NLP dataset**, which consists of parallel translation data for various language pairs. The dataset is sourced from **Hugging Face Datasets** and preprocessed for training.

## Features
- **Self-Attention Mechanism:** Implements scaled dot-product attention.
- **Multi-Head Attention:** Enhances representation learning.
- **Positional Encoding:** Adds sequence information to token embeddings.
- **Encoder-Decoder Architecture:** Fully follows the original Transformer structure.
- **Custom Training Pipeline:** Implements loss functions, optimization, and learning rate scheduling.

## Installation & Requirements
To run the implementation, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage
1. Clone this repository:

   ```sh
   git clone https://github.com/Kousei14/Building-a-Transformer-from-scratch-using-Pytorch.git
   cd Building-a-Transformer-from-scratch-using-Pytorch
   ```

2. Train the model:

   ```sh
   python train.py
   ```

## Future Improvements
- Experimenting with different hyperparameters for better performance.
- Extending the model for additional NLP tasks beyond translation.
- Optimizing for deployment using ONNX or TensorRT.

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is licensed under the MIT License.

---

