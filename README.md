# Python Code Generator AI

A custom-built AI model that generates Python code from natural language descriptions, without relying on pre-trained models or external APIs.

## Features

- Generate Python code from natural language descriptions
- Custom transformer-based architecture
- Built-in tokenizer for Python code
- Code completion suggestions
- Error detection and correction

## Project Structure

```
my_code_generator_ai/
├── data/
│   ├── raw_data/            # Store your Python code dataset here
│   └── processed_data/      # Preprocessed and tokenized data
├── model/
│   ├── transformer.py       # Custom transformer implementation
│   ├── train.py            # Training script
│   └── inference.py        # Code generation script
├── scripts/
│   └── preprocess_data.py  # Data preprocessing script
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/python-code-generator-ai.git
cd python-code-generator-ai
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

## Usage

1. Prepare your dataset:
   - Place your Python code files in `data/raw_data/`
   - Run the preprocessing script:
   ```bash
   python scripts/preprocess_data.py
   ```

2. Train the model:
```bash
python model/train.py
```

3. Generate code:
```python
from model.inference import CodeGenerator

generator = CodeGenerator(
    model_path='checkpoints/best_model.pt',
    tokenizer_path='data/processed_data/code_tokenizer.json'
)

prompt = "Write a function to calculate the factorial of a number"
code = generator.generate_code(prompt)
print(code)
```

## Model Architecture

The code generator uses a custom transformer architecture with:
- Separate encoder and decoder stacks
- Multi-head attention mechanism
- Positional encoding
- Custom tokenizer trained on Python code

## Training Data

The model requires a dataset of Python code files with:
- Function definitions with docstrings
- Class definitions with docstrings
- Clean, well-documented code following PEP 8 guidelines

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
