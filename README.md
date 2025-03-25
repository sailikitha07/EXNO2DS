# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT

## DEVELOPED BY: SAI LIKITHA
## REGISTER NO: 212224230046

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df
```
![alt text](<Screenshot 2025-03-25 110054.png>)


```python
df.info()
```
![alt text](<Screenshot 2025-03-25 110231.png>)

```python
df.shape 
```
![alt text](<Screenshot 2025-03-25 110414.png>)

```python
df.set_index("PassengerId",inplace=True)
df.describe() 
```
![alt text](<Screenshot 2025-03-25 110620.png>)


```python
df.shape
```
![alt text](<Screenshot 2025-03-25 110717.png>)


## Categorical data analysis

```python
df.nunique()
```
![alt text](<Screenshot 2025-03-25 110808.png>)

```python
df["Survived"].value_counts()
```
![alt text](<Screenshot 2025-03-25 110858.png>)

```python
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![alt text](<Screenshot 2025-03-25 110947.png>)


```python
sns.countplot(data=df,x="Survived")
```
![alt text](<Screenshot 2025-03-25 111033.png>)

```python
df.Pclass.unique()
```
![alt text](<Screenshot 2025-03-25 111121.png>)


## Bivariate Analysis

```python
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![alt text](<Screenshot 2025-03-25 111407.png>)

```python
colors = ["blue", "orange"] 
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7,color='violet')
```
![Screenshot 2025-03-25 134940](https://github.com/user-attachments/assets/e4d7c7c2-844e-4910-a8ee-ccaab1aadf52)

```python
colors = ["red", "yellow"]
sns.catplot(x="Survived",hue="Gender",data=df,kind="count",palette=colors)

```
![alt text](<Screenshot 2025-03-25 111600.png>)

```python
df.boxplot(column="Age",by="Survived")
```
![alt text](<Screenshot 2025-03-25 111703.png>)

```python
sns.scatterplot(x=df["Age"],y=df["Fare"])
```

![alt text](<Screenshot 2025-03-25 111749-1.png>)

```python
sns.jointplot(x="Age",y="Fare",data=df)
```
![Screenshot 2025-03-25 140523](https://github.com/user-attachments/assets/dfd9381a-8a60-433e-a858-68356c5d520e)

## Multivariate Analysis

```python
fig,ax1=plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1, x='Pclass', y='Age', hue='Gender', data=df)
plt.show()
```
![alt text](<Screenshot 2025-03-25 112138.png>)

```python
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![alt text](<Screenshot 2025-03-25 112225.png>)

## Co-relation

```python
corr = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr,annot=True)
```
![alt text](<Screenshot 2025-03-25 112354.png>)

```python
sns.pairplot(df)
```
![alt text](<Screenshot 2025-03-25 112528.png>)


# RESULT
We have performed Exploratory Data Analysis on the given data set successfully.

