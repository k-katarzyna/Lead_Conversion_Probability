## Introduction
In today's digital banking landscape, converting leads into customers poses a significant challenge. Leads are sourced through various channels, including online searches, display advertisements, email campaigns, and affiliate partnerships. Happy Customer Bank faces a similar challenge, aiming to achieve a higher conversion rate for a specific loan product.

## The dataset
The dataset contains information about leads, including details such as gender, date of birth, monthly income, employer name, loan amount, interaction data and more. A detailed description can be found in the "Project_main" notebook or by following [this link](https://discuss.analyticsvidhya.com/t/hackathon-3-x-predict-customer-worth-for-happy-customer-bank/3802).

Several challenges are present, including:
* highly imbalanced classes,
* a large number of missing values,
* various data types,
* some variables with unknown meaning.

## The goal
The primary task involves **predicting the probability of loan disbursal**, with the evaluation metric being **ROC AUC**. Narrowing down potential customers to a smaller group with a high probability of loan approval enables the bank to operate more efficiently in increasing its conversion rate. 

The additional step is evaluation of classification skills of the tested models using different discrimination thresholds. Although the choice of threshold can be influenced by various business factors, the secondary objective is to identify the optimal threshold as a reference point.

## The content of the project
The project progress, results and comments can be found in the notebook "Project-main." Custom project utilities are located in the "src" module. Due to the extended duration of some experiments, their results were saved in the "results_data" folder.

To recreate the environment using the included YAML file:
```bash
conda env create -f environment.yml
```
To activate the environment:
```bash
conda activate happy_customer_bank
```
To create the Jupyter environment kernel:
```bash
python -m ipykernel install --user --name=happy_customer_bank
```

## Summary and conclusion
The project involved an extensive exploration of various aspects, including data preprocessing, class balancing, hyperparameter tuning and model selection.

Following the initial testing phase, four models were identified, demonstrating similar behavior and potential effectiveness and suitability for the problem at hand. The highest achieved ROC AUC score, reaching 0.85, was attained by the HistGradientBoostingClassifier. The final results can be seen below.

![Roc curves](Happy_Customer_Bank/results_data/images/roc_curves.png)

This solution serves as a promising starting point for further efforts to enhance the customer conversion rate at the bank, enabling the identification of customer segments with a high probability of loan approval. However, it should be noted that the developed models may not be suitable for precise classification, regardless of the chosen discrimination threshold.

The conducted experiments represent just the tip of the iceberg in addressing this problem. Since the focus was primarily on tree-based ensembles, there is room to explore other types of models along with different data transformations. This opens up exciting possibilities for further research.