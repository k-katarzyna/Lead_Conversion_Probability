import pandas as pd
import numpy as np

# based on: https://www.loomsolar.com/blogs/collections/list-of-cities-in-india
biggest_cities = ["Bangalore", "Delhi", "Chennai", "Hyderabad", "Mumbai", "Pune", "Kolkata", "Ahmedabad"]


def data_preparing_v1(data):
    """
    Preparing the data. The return is the preprocessed DataFrame with modified and transformed columns.
    """
    data = data.copy()

    data.drop("ID", axis=1, inplace=True)
    data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]] = data[["Loan_Amount_Applied", "Loan_Tenure_Applied", "Existing_EMI"]].fillna(0)

    data["City"] = data["City"].apply(lambda x: "Y" if x in biggest_cities else "N" if pd.notna(x) else x)

    data["DOB"] = data["DOB"].apply(lambda x: int(x[-2:]))

    data["Lead_Creation_Date"] = pd.to_datetime(data["Lead_Creation_Date"])
    data["Lead_Creation_Date"] = data["Lead_Creation_Date"].apply(lambda x: x.day_of_year)

    data["Salary_Account"] = data["Salary_Account"].apply(lambda x: 1 if pd.notna(x) else 0)
    data["Employer_Name"] = data["Employer_Name"].apply(lambda x: 0 if pd.isna(x) or (str(x).isdigit()) else 1)

    for feature, value in zip(["Var1", "Var2", "Source"], [1000] * 3):
        rare_values = data[feature].value_counts()[data[feature].value_counts() < value].index.tolist()
        data[feature].replace(rare_values, "Others", inplace=True)

    data.rename(columns={"DOB": "Year_Of_Birth",
                         "Lead_Creation_Date": "Lead_Creation_Day",
                         "City": "Is_Big_City",
                         "Salary_Account": "Account_Provided",
                         "Employer_Name": "Employer_Provided"},
                inplace=True)

    return data