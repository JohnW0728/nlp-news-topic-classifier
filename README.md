# News Topic Classifier: Fine-Tuning RoBERTa vs TF-IDF Baseline

Group project for **WID3002 Natural Language Processing**, Universiti Malaya (Semester 2, Session 2025/2026).

Classifies news articles into 18 topics. A fine-tuned RoBERTa-base model is compared against a classic TF-IDF + Logistic Regression baseline.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 64.8% | 0.65 |
| RoBERTa-base (fine-tuned) | 76.4% | 0.76 |

![Results]
<img width="2580" height="392" alt="image" src="https://github.com/user-attachments/assets/5eddca65-fff9-4e12-8900-3fb992794713" />


## Dataset

Source: the News Category Dataset (v3) by Rishabh Misra, published on Kaggle: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Contents: 209,527 HuffPost headlines (2012-2022) with short descriptions, links, authors, dates and 42 original categories. File used: News_Category_Dataset_v3.json.
Model input: headline and short description combined into one text.
Preprocessing: merged overlapping categories and removed ambiguous ones, giving 18 topics.
Final set: 17,972 samples, capped at 1,000 per topic to balance the classes.
Citation: Misra, R. (2022). News Category Dataset. arXiv:2209.11429. Please check the Kaggle page for the current license terms.
Split: stratified 70% train / 10% validation / 20% test.

## Approach

1. **Baseline:** TF-IDF features with Logistic Regression.
2. **Main model:** RoBERTa-base fine-tuned with PyTorch and Hugging Face Transformers.
3. **Fine-tuning techniques:** layer freezing, layer-wise learning rates, label smoothing and early stopping.
4. **Confidence handling:** uses prediction confidence to flag uncertain or multi-topic articles.
5. **Demo:** a Streamlit app for trying the classifier on new text.

## My contribution

- Fine-tuned the RoBERTa model, improving accuracy from around 60% to over 75%, using layer freezing, layer-wise learning rates, label smoothing and early stopping.
- Reduced and merged overlapping topic categories to make the classes clearer.
- Engineered features to improve accuracy.

## Team

Group 13: John(me), Sean, Penny, Valencien, Yijing, Sanjeevan. The Streamlit demo was built by Sanjeevan.

## How to run

1. Clone the repository and install the libraries: `pip install -r requirements.txt`
2. Download the dataset from the link above and place it in a `data/` folder (not included in this repository).
3. Open `notebooks/NLP_Project_Group_13.ipynb` in Jupyter or Google Colab and run all cells from top to bottom. A GPU is recommended for the RoBERTa training.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── NLP_Project_Group_13.ipynb
└── images/
    └── results.png
```

## Limitations

- Some topics overlap in meaning (for example, similar news categories), which limits accuracy.
- Results come from a single dataset and one train/test split.
