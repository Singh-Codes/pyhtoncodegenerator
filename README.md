# Python Code Generator AI

A custom-built AI model that generates Python code from natural language descriptions, using transformer architecture.

## Status

⚠️ **Note: This model needs to be trained before use**
- The architecture and pipeline are implemented
- Training code is ready
- Requires training on code dataset before generating code

## Features

- Generate Python code from natural language descriptions
- Custom transformer-based architecture
- Built-in tokenizer for Python code
- Code completion suggestions
- Error detection and correction
- Data augmentation
- Performance metrics tracking

## Project Structure

```
pythoncodegenerator/
├── data/
│   ├── raw_data/            # Store your Python code dataset here
│   └── test/                # Test data and examples
├── model/
│   ├── transformer.py       # Custom transformer implementation
│   ├── train.py            # Training script
│   └── inference.py        # Code generation script
├── utils/
│   ├── augmentation.py     # Data augmentation utilities
│   └── metrics.py          # Performance metrics
├── error_resolution/       # Error handling system
├── examples/              # Example code and usage
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Singh-Codes/pyhtoncodegenerator.git
cd pyhtoncodegenerator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

1. Local Training:
```bash
python train.py
```

2. Google Colab Training (Recommended):
- Upload project to Google Drive
- Open and run training notebook
- Use GPU runtime for faster training

## Usage (After Training)

```python
from model.inference import CodeGenerator

generator = CodeGenerator(model_path='checkpoints/best_model.pt')
prompt = "Write a Python function to calculate factorial"
code = generator.generate_code(prompt)
print(code)
```

## Model Architecture

- Transformer-based architecture with:
  - Multi-head attention
  - Positional encoding
  - Custom tokenizer
  - GPU support

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## Author

Singh-Codes
