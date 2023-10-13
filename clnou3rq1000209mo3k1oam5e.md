---
title: "Exploratory Data Analysis using Data Visualization Techniques"
datePublished: Fri Oct 13 2023 16:39:52 GMT+0000 (Coordinated Universal Time)
cuid: clnou3rq1000209mo3k1oam5e
slug: exploratory-data-analysis-using-data-visualization-techniques
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1697204116941/9464a1c1-ee32-4cd2-919b-692032bce39c.jpeg
tags: data-science, data-visualization, exploratory-data-analysis

---

### *<mark>Definition</mark>*

Exploratory Data Analysis (EDA) refers to the process of using statistical and visualization techniques to come up with important aspects of the data for further analysis.

### *<mark>Reasons for EDA</mark>*

* To identify outliers, irrelevant data, and missing values.
    
* To avoid creating an inaccurate training model.
    
* To avoid creating an accurate model with the wrong data.
    

Let's take an example of creating a model that predicts the survival rate of breast cancer patients post-operation with research conducted at the University of Chicago between 1958-1970:

The features that will be included in the model are:

1. Patient’s age at the time of operation (numerical).
    
2. Year of operation (year — 1900, numerical).
    
3. A number of positive axillary nodes were detected (numerical).
    
4. Survival status (class attribute)  
    1: the patient survived 5 years or longer post-operation.  
    2: the patient died within 5 years post-operation
    

The steps to take for our EDA are:

I) Importing libraries and loading data

II) Understand the data (click the replit link below for code and further observation):

%[https://replit.com/@NivethaB2/edaandvisualization#main.py] 

As it can be seen, the dataset contains 305 rows and 4 columns.

Using df\['survival\_status'\].value\_counts(), the output shows:

1 224

2 81 meaning

Out of a total of 305 patients, the number of patients who survived over 5 years post-operation is nearly 3 times the number of patients who died within 5 years.

### *<mark>Data Preparation</mark>*

The original class labels — 1 (survived 5 years and above) and 2 (died within 5 years) are not in accordance with the case.

we map survival status values 1 and 2 in the column *survival\_status* to categorical variables ‘yes’ and ‘no’ respectively such that,  
survival\_status = 1 → survival\_status = ‘yes’  
survival\_status = 2 → survival\_status = ‘no’

```python
df['survival_status'] = df['survival_status'].map({1:"yes", 2:"no"})
```

### *<mark>General statistical analysis</mark>*

* On average, patients got operated at the age of 63.
    
* An average number of positive axillary nodes detected = 4.
    
* As indicated by the 50th percentile, the median of positive axillary nodes is 1.
    
* As indicated by the 75th percentile, 75% of the patients have less than 4 nodes detected.
    

If you see, there is a significant difference between the mean and the median values. This is because there are some outliers in our data and the mean is influenced by the presence of outliers.

### *<mark>Uni-variate Data Analysis</mark>*

This kind of analysis is done by considering one variable at a time.

Let’s say our aim is to be able to correctly determine the survival status given the features — patient’s age, operation year, and positive axillary node count. Which among these 3 variables is more useful than other variables in order to distinguish between the class labels ‘yes’ and ‘no’? To answer this, we’ll plot the distribution plots (also called probability density functions or PDF plots) with each feature as a variable on *X*\-axis. The values on the Y-axis in each case represent the normalized density.

### *<mark>Distribution Plots</mark>*

**1\. Patient’s age**

```python
sns.FacetGrid(df, hue = "survival_status").map(sns.distplot, "patient_age").add_legend()
plt.show()
```

* Among all the age groups, the patients belonging to 40-60 years of age are the highest.
    
* There is a high overlap between the class labels. This implies that the survival status of the patient post-operation cannot be discerned from the patient’s age.
    

**2\. Operation year**

```python
sns.FacetGrid(df, hue = "survival_status").map(sns.distplot, "operation_year").add_legend()
plt.show()
```

There is a huge overlap between the class labels suggesting that one cannot make any distinctive conclusion regarding the survival status based solely on the operation year.

**3\. Number of positive axillary nodes**

```python
sns.FacetGrid(df, hue = "survival_status").map(sns.distplot, "positive_axillary_nodes").add_legend()
plt.show()
```

### *<mark>Box plots and Violin plots</mark>*

Box plots display data in 5 numbers; minimum, lower quartile(25th percentile), median(50th percentile), upper quartile(75th percentile), and maximum data values.

Below is an example of a box plot:

```python
plt.figure(figsize = (15, 4))
plt.subplot(1,3,1)
sns.boxplot(x = 'survival_status', y = 'patient_age', data = df)
plt.subplot(1,3,2)
sns.boxplot(x = 'survival_status', y = 'operation_year', data = df)
plt.subplot(1,3,3)
sns.boxplot(x = 'survival_status', y = 'positive_axillary_nodes', data = df)
plt.show()
```

Violin plots are more informative as compared to box plots as violin plots also represent the underlying distribution of the data. Below shows a violin plot:

```python
plt.figure(figsize = (15, 4))
plt.subplot(1,3,1)
sns.violinplot(x = 'survival_status', y = 'patient_age', data = df)
plt.subplot(1,3,2)
sns.violinplot(x = 'survival_status', y = 'operation_year', data = df)
plt.subplot(1,3,3)
sns.violinplot(x = 'survival_status', y = 'positive_axillary_nodes', data = df)
plt.show()
```

### ***<mark>Bi-variate data analysis</mark>***

### *<mark>Pair plot</mark>*

We'll plot a pair plot to visualize the relationship between the features in a pairwise manner.

```python
sns.set_style('whitegrid')
sns.pairplot(df, hue = 'survival_status')
plt.show()
```

### ***<mark>Joint plot</mark>***

It provides bi-variate plots with uni-variate marginal distributions.

```python
sns.jointplot(x = 'patient_age', y = 'positive_axillary_nodes', data = df)
plt.show()
```

### *<mark>Heatmap</mark>*

It's used to obtain the feature importance in regression analysis.

Although correlated features do not impact the performance of the statistical model, it could mess up the post-modeling analysis.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'df' is your DataFrame

# Generate the heatmap
sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)

# Show the plot
plt.show()
```

### *<mark>Multivariate analysis with Contour plot</mark>*

A graphical technique for representing a 3-dimensional surface by plotting constant *z* slices, called contours, in a 2-dimensional format. A contour plot enables us to visualize data in a two-dimensional plot.

```python
sns.jointplot(x = 'patient_age',  y = 'operation_year' , data = df,  kind = 'kde', fill = True)
plt.show()
```

**Article courtesy of** [**Analytics Vidya**](https://www.analyticsvidhya.com/blog/2021/08/exploratory-data-analysis-and-visualization-techniques-in-data-science/#h-why-exploratory-data-analysis-is-important)