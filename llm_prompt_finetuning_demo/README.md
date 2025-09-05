LLM Prompt Engineering + Fine-Tuning Demo

This project demonstrates two fundamental areas for modern AI engineers:

Prompt Engineering: Using techniques such as Chain-of-Thought and Instruction Prompting.

Fine-Tuning: Performing parameter-efficient fine-tuning of a language model using a custom dataset with HuggingFace.

📁 Project Structure

notebooks/prompt_engineering.ipynb — Practical demonstration of prompt construction and testing.

scripts/fine_tuning.py — Fine-tuning script using HuggingFace Transformers and Trainer API.

data/sample_data.json — Small dataset for fine-tuning simulation.

⚙️ How to Use

Install the dependencies:

pip install transformers datasets accelerate


Run the fine-tuning:

python scripts/fine_tuning.py


Explore the prompt engineering notebook with Jupyter.

🔍 Notes

This project uses lightweight models such as distilbert-base-uncased, which can run on CPU (but slower).
