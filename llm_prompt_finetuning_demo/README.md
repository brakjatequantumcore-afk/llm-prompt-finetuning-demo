# LLM Prompt Engineering + Fine-Tuning Demo

Este projeto demonstra duas áreas fundamentais para engenheiros de IA modernos:
1. **Prompt Engineering**: Uso de técnicas como *Chain-of-Thought* e *Instruction Prompting*.
2. **Fine-Tuning**: Ajuste fino de um modelo de linguagem utilizando um dataset customizado com HuggingFace.

## Estrutura do Projeto

- `notebooks/prompt_engineering.ipynb` — Demonstração prática de construção e teste de prompts.
- `scripts/fine_tuning.py` — Script de fine-tuning usando HuggingFace Transformers e Trainer API.
- `data/sample_data.json` — Dataset pequeno para simulação de treinamento.

## Como usar

1. Instale as dependências:
```bash
pip install transformers datasets accelerate
```

2. Execute o fine-tuning:
```bash
python scripts/fine_tuning.py
```

3. Explore o notebook de prompt engineering no Jupyter.

## Observação

Este projeto roda com modelos pequenos como `distilbert-base-uncased` e pode ser executado em CPU (mais lento).