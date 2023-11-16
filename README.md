### Introduction
In today's digital banking landscape, converting leads into customers poses a significant challenge. Leads are sourced through various channels, including online searches, display advertisements, email campaigns, and affiliate partnerships. Happy Customer Bank faces a similar challenge, aiming to achieve a higher conversion rate for a specific loan product. More details about the task and the dataset can be found [here](https://discuss.analyticsvidhya.com/t/hackathon-3-x-predict-customer-worth-for-happy-customer-bank/3802).

### The dataset
The dataset contains leads' information, covering details like gender, date of birth, monthly income, employer name, loan amount, interaction data and more.

Several challenges are present, including:
* highly imbalanced classes,
* a large number of missing values,
* various data types,
* some variables with unknown meanings.

### The goal
The primary task involves **predicting the probability of loan disbursal**, with the evaluation metric being **ROC_AUC**. Narrowing down potential customers to a smaller group with a high probability of loan approval will enable the bank to operate more efficiently in increasing its conversion rate.

As a subsequent step, the classification skills of the tested models were also determined by applying different discrimination thresholds.

### The content of the project
The project progress, results and comments can be found in the notebook "Project-main." Custom project utilities are located in the "src" module. Due to the extended duration of some experiments, their results were saved in the "results_data" folder.

### Setting up the project environment
To recreate environment using the included YAML file:
```bash
conda env create -f environment.yml
```
To activate the environment:
```bash
conda activate happy_customer_bank
```
To create the environment Jupyter kernel:
```bash
python -m ipykernel install --user --name=happy_customer_bank
```




