# Legal-Text-Summarization-Using-Transformers
Repository for our NLP Project during Fall '24 on Legal Text Summarization Using Transformers.

![Overall Architecture](https://github.com/shravan-18/Legal-Text-Summarization-Using-Transformers/blob/main/Paper-Resources/Overall-Architecture.png)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Adaptive Multi-Head Attention](#adaptive-multi-head-attention)
  - [Training and Fine-Tuning](#training-and-fine-tuning)
  - [Summarization Process](#summarization-process)
- [Results](#results)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction

Legal documents are often lengthy and complex, making it challenging for individuals to grasp essential information quickly. Summarization of legal texts can significantly aid in understanding and decision-making processes. This project focuses on developing an efficient and effective method for summarizing legal documents using Transformer-based models with an innovative **Adaptive Multi-Head Attention** mechanism.

## Project Overview

This project implements a custom Transformer model integrated with an Adaptive Multi-Head Attention mechanism to summarize legal texts. The approach combines the strengths of pre-trained models with a novel attention mechanism to enhance performance on legal document summarization tasks.

The project structure includes:

- Reading and preprocessing legal documents from PDF files.
- Implementing a custom Transformer model with Adaptive Multi-Head Attention.
- Fine-tuning the model on a dataset of legal documents and summaries.
- Generating summaries for new legal documents.

## Motivation

The legal domain deals with extensive documents containing intricate details. Manually summarizing these documents is time-consuming and prone to errors. Automating this process can:

- Increase efficiency in legal workflows.
- Provide quick insights into lengthy contracts, agreements, and policies.
- Assist legal professionals and laypersons in understanding complex legal jargon.

## Methodology

### Data Collection

A dataset of legal documents and their corresponding summaries is essential for training the model. The dataset encompasses various types of legal texts to ensure the model generalizes well.

### Data Preprocessing

- **PDF Extraction**: Legal documents in PDF format are read using the `PyPDF2` library.
- **Text Normalization**: Extracted text is normalized by converting to lowercase, removing special characters, and handling whitespace.
- **Tokenization**: The text is tokenized using the `BartTokenizer` from Hugging Face's Transformers library.
- **Encoding**: Tokenized inputs and summaries are encoded into numerical representations suitable for the model.

### Model Architecture

#### Transformer Model

The base model is a Transformer-based architecture, specifically leveraging the BART model pre-trained for sequence-to-sequence tasks. BART is chosen for its effectiveness in text generation and summarization tasks.

#### Adaptive Multi-Head Attention

A novel **Adaptive Multi-Head Attention** mechanism is integrated into the Transformer model. This mechanism dynamically adjusts the number of attention heads based on the input, aiming to improve the model's ability to focus on relevant parts of the text.

Key aspects of the Adaptive Multi-Head Attention:

- **Dynamic Head Selection**: The mechanism selects active attention heads adaptively during runtime.
- **Compatibility**: It maintains compatibility with the pre-trained BART model by ensuring output dimensions match expected values.
- **Innovation**: Introduces novelty to the standard attention mechanism, making the project unique.

### Training and Fine-Tuning

- **Loading Pre-trained Weights**: The model loads pre-trained weights from the BART base model, excluding the attention layers, which are re-initialized.
- **Optimizer and Scheduler**: Uses `AdamW` optimizer with a learning rate scheduler for efficient training.
- **Loss Function**: The model computes the cross-entropy loss between the generated summaries and the reference summaries.
- **Training Loop**: Iteratively updates the model weights using backpropagation over multiple epochs.

### Summarization Process

- **Inference Mode**: The trained model is set to evaluation mode for generating summaries.
- **Input Processing**: New legal documents are processed similarly to the training data.
- **Generation Parameters**: Parameters like `max_length`, `min_length`, `num_beams`, and `length_penalty` are set to control the summary length and quality.
- **Output Decoding**: The generated token IDs are decoded back into text using the tokenizer.

## Results

The model demonstrates the ability to generate coherent summaries of legal documents. The Adaptive Multi-Head Attention mechanism contributes to focusing on important sections of the text, potentially improving summarization quality.

**Note**: Quantitative results such as ROUGE scores, BLEU scores, or human evaluation metrics should be included here based on the actual evaluation conducted during the project.

## Usage

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Transformers library (Hugging Face)
- PyPDF2
- Other dependencies as listed in the `requirements.txt` file.

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/shravan-18/Legal-Text-Summarization-Using-Transformers.git
   ```

2. **Change Directory:**:

   ```bash
   cd Legal-Text-Summarization-Using-Transformers
   ```

3. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook

The project's implementation is detailed in the Jupyter notebook available at:

`Notebook.ipynb`

To run the notebook:

1. **Start Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**:

   ```bash
   Navigate to the `Notebook.ipynb` file in the Jupyter interface.
   ```

3. **Run the Cells:**:

   Execute the cells sequentially to reproduce the results. Ensure that you have the necessary legal documents in PDF format placed in the appropriate directory as specified in the notebook.

### Conclusion

This project presents a novel approach to legal text summarization by integrating an Adaptive Multi-Head Attention mechanism into a Transformer-based model. The method leverages the strengths of pre-trained models while introducing innovation in the attention mechanism to enhance performance on domain-specific tasks.

### References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [PyPDF2 Documentation](https://pypi.org/project/PyPDF2/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
