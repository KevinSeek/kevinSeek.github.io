---
layout: post
title:  "SAT & ACT Competitor Analysis"
date:   2020-10-21 22:00:00 +0800
categories: [Python, Data-Cleaning]
---


# SAT & ACT Competitor Analysis

## Introduction

## Problem Statement

The new format for the SAT was released in March 2016. As an employee of the College Board - the organization that administers the SAT - <b style="color:magenta">you are a part of a team that tracks statewide participation and recommends where money is best spent to improve SAT participation rates.</b>

## Executive Summary

<details><summary>Summary of findings</summary>

SAT & ACT are the 2 standardized test used by many colleges in US as part of their admission decision process. Juniors and seniors in high school usually take these tests to demonstrate their readiness for college level work. Both tests are designed to different way to measure college readiness and predict future academic success.

After a new format of SAT was released, we would like to see how it perform against it's competitor - ACT in term of popularity (measured by participation rate) among high school students as the standardized test for their college admission.

We shall limit the scope of our analysis to states that have SAT participation rate in 2018 that are more than 50%. Other factors may be in play for states which have less 50% participation rate, eg. strong political affiliation to ACT and more data will be necessary which is beyond the scope of this project.


Key Observations
    <ul>
        <li>SAT & ACT participation rates in each states are mostly inversely related. An increase in one often mean a decrease in another.</li>
        <li>Negative correlation between participation rate and respective test scores for both SAT & ACT</li>
        <li>Year-on-year participation in both states tends to remain the same. Exception for Illinois & Colorado where there are policy changes.</li>
        <li>Preference for one test over the other maybe geographically driven - coastal states preferring SAT over ACT.</li>    
    </ul>


Additional observation
        <ul>
        <li>We discover states such as Illinois & Colorado have almost a 90% jump in participation rate in 2018 compared to 2017. This may be due to the new format of SAT and various new initiatives such as SAT 'school day' event and educational investment by state.</li>
        <li>States which do not have specific preferred test (states that have more than 50% participation rate in both SAT & ACT) also see a marginal increase in SAT participation rate in 2018.</li>  
    </ul>


Based on analysis of the data & additional research, I choose Florida as the next state we can invest in. The reason for choosing the state are as follow:
    <ol>
        <li>Florida is the 3rd largest state in term of population which means that there are likely to have more students participating in college admission test.</li>
        <li>As an investment risk mitigation option, we may not be able to replicate the success case of Illinois & Colorado, low SAT to high SAT participation rate. Using Florida, which already have close to 50% participation rate in both SAT and ACT, it is likely to have less 'barrier to entry' for students to switch from ACT to SAT since they may be familiar with the structure of both exam.</li>
        <li>Florida is committed to make big investment into education, hence it is highly likely to win political favor in expanding into Florida, given that our part of our current initiatives (free online review/classes) can contribute to more students being admitted to college.</li>
    </ol>

In addition, I believe that we should continue & expand current initiatives to other states which still show weakness in SAT participation rate (>50% but below 75%) based on previous year (2018). This way, we are likely to see improvement in participation rate year-on-year across all states going forward.</details>

## Contents:

- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [2017 Data Import & Cleaning](#Data-Import-and-Cleaning)
- [2018 Data Import and Cleaning](#2018-Data-Import-and-Cleaning)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Data Visualization](#Visualize-the-data)
- [Descriptive and Inferential Statistics](#Descriptive-and-Inferential-Statistics)
- [Outside Research](#Outside-Research)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)

# Code Book

## Import Library


```python
#Imports:
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.width", 120)
```


```python
# Optional setting to setup Jupyter
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 80%; }
    div#menubar-container     { width: 80%; }
    div#maintoolbar-container { width: 80%; }
</style>
"""))
```



<style>
    div#notebook-container    { width: 80%; }
    div#menubar-container     { width: 80%; }
    div#maintoolbar-container { width: 80%; }
</style>



## User Defined Functions


```python
# Create User-Defined Functions
def subplot_regplot(x, y, df, axes, title, ci=None):
    '''
     Plot a regplot with subplot titles

     Parameters
     ----------
       x (str): Dataframe column on x-axis
       y (str): Dataframe column on y-axis
       df (df): Dataframe to plot on
       axes (tup): ax tuple
       title (str): subplot title
       ci (bool): show confidence level

     Returns:
     ----------
       None    
    '''
    sns.regplot(x=x, y=y, data=df, ax=axes, ci=ci, line_kws={'color':'purple'})
    axes.set_title(title)
    return None
```


```python
def plot_swarmbox(dataframe, list_of_columns, ax_title, ax_ylabel, ax_xlabel, orient='v', figsize=(10,10)):
    '''
    Plot a combo chart of boxplot & Swarmplot of dataframe

    Parameters
    ----------
       dataframee      (df): Dataframe to plot on
       list_of_columns (list): list of columns to plot on
       ax_title        (str): subplot title
       ax_ylabel       (str): y label
       ax_xlabel       (str): x label
       orient          (str): v for vertical orientation | h for horizontal orientation
       figsize         (tup): tuple to set canvas size

    Returns:
    ----------
       None     
    '''

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Boxplot
    ## Change the boxplot properties
    sns.boxplot(data=dataframe[list_of_columns], orient=orient,
                    showcaps=True,capprops={'color':'red', 'linewidth': 2} ,boxprops={'facecolor':'None','edgecolor':'lightseagreen'},
                    showfliers=False,whiskerprops={'linewidth':0}, medianprops={'color':'lightseagreen'}, ax=ax)

    # Swarmplot
    sns.swarmplot(data=dataframe[list_of_columns], orient=orient, zorder=1, ax=ax)

    ax.set_title(ax_title)
    ax.set_ylabel(ax_ylabel)
    ax.set_xlabel(ax_xlabel)
    #ax.grid(color='blue', linestyle='-.', linewidth=1)
    return None
```


```python
def plot_boxplt(dataframe, list_of_columns, ax_title, ax_ylabel, ax_xlabel, orient='v', figsize=(10,10)):
    '''
    Plot a boxplot of dataframe

    Parameters
    ----------
       dataframee      (df): Dataframe to plot on
       list_of_columns (list): list of columns to plot on
       ax_title        (str): subplot title
       ax_ylabel       (str): y label
       ax_xlabel       (str): x label
       orient          (str): v for vertical orientation | h for horizontal orientation
       figsize         (tup): tuple to set canvas size

    Returns:
    ----------
       None     
    '''

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.boxplot(data=dataframe[list_of_columns], orient=orient)
    ax.set_title(ax_title)
    ax.set_ylabel(ax_ylabel)
    ax.set_xlabel(ax_xlabel)
    ax.grid(color='blue', linestyle='-.', linewidth=1)
    return None
```


```python
def melter(df1,col1, df2, col2, var_name, val_name):
    '''
     Melt 2 dataframes based on column and append them together

     Parameters
     ----------
       df1        (str): 1st dataframe
       col1       (str): 2nd dataframe
       df2        (str): 1st dataframe col      
       col2       (str): 2nd dataframe col
       var_name   (str): common variable col name
       val_name   (str): common value col name

     Returns:
     ----------
       return an appended metled dataframe

   '''
    df_1 = df1[[col1]].melt(var_name=var_name, value_name=val_name)
    df_2 = df2[[col2]].melt(var_name=var_name, value_name=val_name)
    df = df_1.append(df_2)
    df.reset_index(drop=True, inplace=True)
    return df
```


```python
# Create User-Defined Functions
def gridplot(df, col1, col2, df2, df2_col1, df2_col2, df2_col3, fs_tuple, fig_title, sp1_title, sp2_title, sp3_title):
    '''
     plot a distplot, swarm/box and a scatter plot of the 2 col of dataframe

     Parameters
     ----------
       df1        (str): 1st dataframe
       col1       (str): 1st dataframe col1
       col2       (str): 1st dataframe col2
       df2        (str): 2nd dataframe
       df2_col1   (str): 2nd dataframe col 1
       df2_col2   (str): 2nd dataframe col 2
       df2_col3   (str): 2nd dataframe col 3
       fs_tuple   (tup): Figsize tuple
       fig_title  (str): Canvas title
       sp1_title  (str): subplot 1 title
       sp2_title  (str): subplot 2 title
       sp3_title  (str): subplot 3 title

     Returns:
     ----------
       return None
   '''
    # Define canvas
    fig = plt.figure(figsize=fs_tuple)
    gs = fig.add_gridspec(2, 2,top = 0.85, hspace= 0.2, wspace=0.2)
    plt.title(fig_title, fontsize=20, pad=10)
    plt.axis('off')

    # plot 1st subplot - displot
    f_ax1 = fig.add_subplot(gs[0, :])
    sns.distplot(df[col1],bins=np.arange(0,100,10),kde=True, color='blue', ax=f_ax1)
    sns.distplot(df[col2],bins=np.arange(0,100,10),kde=True, color='red', ax=f_ax1)
    f_ax1.set_title(sp1_title)
    f_ax1.set_ylabel('Density')
    f_ax1.legend([col1,col2])


    # plot 2nd subplot - swarm/box plot
    f_ax2 = fig.add_subplot(gs[1, 0])

    ## Swarmplot
    sns.swarmplot(data=df[[col1,col2]],palette=['Blue','red'], zorder=1, ax=f_ax2)    

    ## Boxplot
    sns.boxplot(data=df[[col1,col2]],
                    showcaps=True,capprops={'color':'orange', 'linewidth': 2} ,boxprops={'facecolor':'None','edgecolor':'lightseagreen'},
                    showfliers=False,whiskerprops={'linewidth':0}, medianprops={'color':'lightseagreen'}, ax=f_ax2)    
    f_ax2.set_title('Swarm/Box plot of SAT Part Rate', fontsize=12)

    # Plot 3rd subplot - scatterplot
    f_ax3 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x=df2_col1, y=df2_col2, data=df2, palette=['Blue','red'],\
                alpha=0.5, hue=df2_col3, ax=f_ax3)
    f_ax3.set_title(sp3_title, fontsize=12)

    return None
```

## Data Import


```python
# 2017 SAT & ACT Dataset
sat_2017 = pd.read_csv('data/sat_2017.csv')
act_2017 = pd.read_csv('data/act_2017.csv')
```


```python
# First 5 rows of SAT 2017 dataset
print(f'shape of sat_2017: {sat_2017.shape}')
sat_2017.head()
```

    shape of sat_2017: (51, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation</th>
      <th>Evidence-Based Reading and Writing</th>
      <th>Math</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5%</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>38%</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>30%</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3%</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>53%</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat_2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51 entries, 0 to 50
    Data columns (total 5 columns):
     #   Column                              Non-Null Count  Dtype
    ---  ------                              --------------  -----
     0   State                               51 non-null     object
     1   Participation                       51 non-null     object
     2   Evidence-Based Reading and Writing  51 non-null     int64
     3   Math                                51 non-null     int64
     4   Total                               51 non-null     int64
    dtypes: int64(3), object(2)
    memory usage: 2.1+ KB



```python
sat_2017.isnull().sum()
```




    State                                 0
    Participation                         0
    Evidence-Based Reading and Writing    0
    Math                                  0
    Total                                 0
    dtype: int64




```python
# First 5 rows of ACT 2017 dataset
print(f'shape of act_2017: {act_2017.shape}')
act_2017.head()
```

    shape of act_2017: (52, 7)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation</th>
      <th>English</th>
      <th>Math</th>
      <th>Reading</th>
      <th>Science</th>
      <th>Composite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>National</td>
      <td>60%</td>
      <td>20.3</td>
      <td>20.7</td>
      <td>21.4</td>
      <td>21.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>100%</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alaska</td>
      <td>65%</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arizona</td>
      <td>62%</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arkansas</td>
      <td>100%</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 52 entries, 0 to 51
    Data columns (total 7 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   State          52 non-null     object
     1   Participation  52 non-null     object
     2   English        52 non-null     float64
     3   Math           52 non-null     float64
     4   Reading        52 non-null     float64
     5   Science        52 non-null     float64
     6   Composite      52 non-null     object
    dtypes: float64(4), object(3)
    memory usage: 3.0+ KB



```python
act_2017.isnull().sum()
```




    State            0
    Participation    0
    English          0
    Math             0
    Reading          0
    Science          0
    Composite        0
    dtype: int64




```python
act_2017['Composite'].unique()
```




    array(['21.0', '19.2', '19.8', '19.7', '19.4', '22.8', '20.8', '25.2',
           '24.1', '24.2', '21.4', '19.0', '22.3', '22.6', '21.9', '21.7',
           '20.0', '19.5', '24.3', '23.6', '25.4', '21.5', '18.6', '20.4',
           '20.3', '17.8', '25.5', '23.9', '19.1', '22.0', '21.8', '23.7',
           '24.0', '18.7', '20.7', '23.8', '20.5', '20.2x'], dtype=object)



<span style = "color:Magenta">Analysis & Comments:</span><br>

SAT
- Has 51 rows and 5 columns.
- There are 2 string data type and 3 numeric data type.
- For columns of numeric data, the datatype is correct. However, participatioin is a wrong datatype. It should be numerical rather than string. This is likely due to the '%' sign behind. Remove the '%' symbol and convert it to numeric will help in data wrangling.

ACT
- Has 52 rows and 7 columns.
- There are 3 string data type and 4 numeric data type
- Participation column is of wrong data type and suffers the same the issue as participation in SAT.
- Composite column is of wrong data type. This is due to a value '20.2x' which causes the entire column to be imputed as string. correctly this will allow the column to transform to numeric type and allow easy data wrangling.

All in all, there is no null values. Comparing SAT and ACT dataset, ACT has an extra row which shows the National Average. This row will be remove as it is an aggregated row.

## Data Cleaning 2017 Dataset

### Convert column headers into lower case


```python
sat_2017.columns = sat_2017.columns.str.lower()
sat_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>evidence-based reading and writing</th>
      <th>math</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5%</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>38%</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>30%</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3%</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>53%</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2017.columns = act_2017.columns.str.lower()
act_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>english</th>
      <th>math</th>
      <th>reading</th>
      <th>science</th>
      <th>composite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>National</td>
      <td>60%</td>
      <td>20.3</td>
      <td>20.7</td>
      <td>21.4</td>
      <td>21.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>100%</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alaska</td>
      <td>65%</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arizona</td>
      <td>62%</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arkansas</td>
      <td>100%</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
  </tbody>
</table>
</div>



### Removal of '%' from Participation


```python
# SAT 2017 dataset
sat_2017['participation'] = sat_2017['participation'].apply(str.replace,args=('%',''))
sat_2017['participation'] = sat_2017['participation'].astype(int)
sat_2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51 entries, 0 to 50
    Data columns (total 5 columns):
     #   Column                              Non-Null Count  Dtype
    ---  ------                              --------------  -----
     0   state                               51 non-null     object
     1   participation                       51 non-null     int32
     2   evidence-based reading and writing  51 non-null     int64
     3   math                                51 non-null     int64
     4   total                               51 non-null     int64
    dtypes: int32(1), int64(3), object(1)
    memory usage: 1.9+ KB



```python
# ACT 2017 dataset
act_2017['participation'] = act_2017['participation'].apply(str.replace,args=('%',''))
act_2017['participation'] = act_2017['participation'].astype(int)
act_2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 52 entries, 0 to 51
    Data columns (total 7 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   state          52 non-null     object
     1   participation  52 non-null     int32  
     2   english        52 non-null     float64
     3   math           52 non-null     float64
     4   reading        52 non-null     float64
     5   science        52 non-null     float64
     6   composite      52 non-null     object
    dtypes: float64(4), int32(1), object(2)
    memory usage: 2.8+ KB


### Removing of errors in ACT 2017 Composite Column


```python
act_2017['composite'] = act_2017['composite'].apply(str.replace, args=('x',''))
act_2017['composite'] = act_2017['composite'].astype(float)
# Remove the aggregated row
act_2017 = act_2017[act_2017['state']!='National']
act_2017.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 51 entries, 1 to 51
    Data columns (total 7 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   state          51 non-null     object
     1   participation  51 non-null     int32  
     2   english        51 non-null     float64
     3   math           51 non-null     float64
     4   reading        51 non-null     float64
     5   science        51 non-null     float64
     6   composite      51 non-null     float64
    dtypes: float64(5), int32(1), object(1)
    memory usage: 3.0+ KB


### Describing Data


```python
sat_2017.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>participation</th>
      <td>51.0</td>
      <td>39.803922</td>
      <td>35.276632</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>66.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>evidence-based reading and writing</th>
      <td>51.0</td>
      <td>569.117647</td>
      <td>45.666901</td>
      <td>482.0</td>
      <td>533.5</td>
      <td>559.0</td>
      <td>613.0</td>
      <td>644.0</td>
    </tr>
    <tr>
      <th>math</th>
      <td>51.0</td>
      <td>547.627451</td>
      <td>84.909119</td>
      <td>52.0</td>
      <td>522.0</td>
      <td>548.0</td>
      <td>599.0</td>
      <td>651.0</td>
    </tr>
    <tr>
      <th>total</th>
      <td>51.0</td>
      <td>1126.098039</td>
      <td>92.494812</td>
      <td>950.0</td>
      <td>1055.5</td>
      <td>1107.0</td>
      <td>1212.0</td>
      <td>1295.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat_2017[sat_2017['participation']==2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>evidence-based reading and writing</th>
      <th>math</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>2</td>
      <td>641</td>
      <td>635</td>
      <td>1275</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>2</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>2</td>
      <td>635</td>
      <td>621</td>
      <td>1256</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat_2017[sat_2017['math']==52]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>evidence-based reading and writing</th>
      <th>math</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Maryland</td>
      <td>69</td>
      <td>536</td>
      <td>52</td>
      <td>1060</td>
    </tr>
  </tbody>
</table>
</div>



<span style = "color:Magenta">Analysis & Comments:</span><br>

SAT
- 800 points per section with a max score of 1600
- Not scaled by test population. Actual Score is used hence possible to see actual Min & Max score.


SAT 2017
1. Participation rate has a mean at 39% while min is at 2% and max at 100% suggest high variance across states. 3 states = Lowa, Mississippi & North Dakota have the lowest participation rate at 2%.
2. Both Writing & Math have a mean of around 540++ and max of around 640+. However, the min for math is 52. Upon further checks, Maryland is the outlier that resulted in a math score of 52.


```python
act_2017.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>participation</th>
      <td>51.0</td>
      <td>65.254902</td>
      <td>32.140842</td>
      <td>8.0</td>
      <td>31.00</td>
      <td>69.0</td>
      <td>100.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>english</th>
      <td>51.0</td>
      <td>20.931373</td>
      <td>2.353677</td>
      <td>16.3</td>
      <td>19.00</td>
      <td>20.7</td>
      <td>23.30</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>math</th>
      <td>51.0</td>
      <td>21.182353</td>
      <td>1.981989</td>
      <td>18.0</td>
      <td>19.40</td>
      <td>20.9</td>
      <td>23.10</td>
      <td>25.3</td>
    </tr>
    <tr>
      <th>reading</th>
      <td>51.0</td>
      <td>22.013725</td>
      <td>2.067271</td>
      <td>18.1</td>
      <td>20.45</td>
      <td>21.8</td>
      <td>24.15</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>science</th>
      <td>51.0</td>
      <td>21.041176</td>
      <td>3.182463</td>
      <td>2.3</td>
      <td>19.90</td>
      <td>21.3</td>
      <td>22.75</td>
      <td>24.9</td>
    </tr>
    <tr>
      <th>composite</th>
      <td>51.0</td>
      <td>21.519608</td>
      <td>2.020695</td>
      <td>17.8</td>
      <td>19.80</td>
      <td>21.4</td>
      <td>23.60</td>
      <td>25.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2017[act_2017['participation']==8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>english</th>
      <th>math</th>
      <th>reading</th>
      <th>science</th>
      <th>composite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Maine</td>
      <td>8</td>
      <td>24.2</td>
      <td>24.0</td>
      <td>24.8</td>
      <td>23.7</td>
      <td>24.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2017[act_2017['science']==2.3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>participation</th>
      <th>english</th>
      <th>math</th>
      <th>reading</th>
      <th>science</th>
      <th>composite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Maryland</td>
      <td>28</td>
      <td>23.3</td>
      <td>23.1</td>
      <td>24.2</td>
      <td>2.3</td>
      <td>23.6</td>
    </tr>
  </tbody>
</table>
</div>



<span style = "color:Magenta">Analysis & Comments:</span><br>

ACT
- 36 points per section with a max score of 36 (Composite is an avg of all sections)
- Scaling by test population. Not possible to see the actual min and max score


ACT 2017
1. Because of the scaling, the mean for each sections are around 20+ range. However, even after scaling, science min score is at 2.3. Again, Maryland is the outlier that resulted in the lowest score for ACT,science.
2. Participation Rate mean is around 65% suggesting a higher part rate compared to SAT. However, Maine has the lowest participation rate at 8%.

### Renaming Columns

By renaming the columns to be more expressive can allow us qucikly to tell the difference between SAT & ACT columns and by using right abbreviation, it reduces the efforts to transform the data yet retaining the meaning behind the columns.


```python
# Create a mapping dictionary for both SAT & ACT
# Revert prefixed states to without prefixs
header_dict =  {'participation': 'part_rate',
                'evidence-based reading and writing': 'ebrw',
                'math': 'math',
                'total': 'total',
                'english': 'eng',
                'Reading': 'read',
                'science': 'sci',
                'composite':'total'}

# Add sat prefix to SAT dataset
sat_2017.rename(header_dict,axis=1, inplace=True)
sat_2017 = sat_2017.add_prefix('sat_')
sat_2017.rename({'sat_state':'state'}, axis=1, inplace=True)
sat_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate</th>
      <th>sat_ebrw</th>
      <th>sat_math</th>
      <th>sat_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>38</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>30</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>53</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add act prefix to ACT dataset
act_2017.rename(header_dict,axis=1, inplace=True)
act_2017 = act_2017.add_prefix('act_')
act_2017.rename({'act_state':'state'}, axis=1, inplace=True)
act_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>act_part_rate</th>
      <th>act_eng</th>
      <th>act_math</th>
      <th>act_reading</th>
      <th>act_sci</th>
      <th>act_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Alabama</td>
      <td>100</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alaska</td>
      <td>65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arizona</td>
      <td>62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arkansas</td>
      <td>100</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>California</td>
      <td>31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>



### Merging DataFrame


```python
# Print the shape of act_2017 & sat_2017
print(f'shape of act_2017 dataframe: {act_2017.shape}')
print(f'shape of sat_2017 dataframe: {sat_2017.shape}')
print('')
print('*' * 60)
print('')
# merge dataframe - common column is state
test_2017 = pd.merge(sat_2017, act_2017, how='outer', on='state')
print(f'shape of test_2017 dataframe: {test_2017.shape}')
```

    shape of act_2017 dataframe: (51, 7)
    shape of sat_2017 dataframe: (51, 5)

    ************************************************************

    shape of test_2017 dataframe: (51, 11)


<span style = "color:Magenta">Analysis & Comments:</span><br>
Given that the states in both dataframes are the same Joining by the key column = 'state' using either 'inner', 'outer', 'left', 'right' will produce the same result. However, to future proof the codes, using 'outer' will be more comprehensive to capture/preserve any new state addition from both dataframes.


```python
test_2017.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate</th>
      <th>sat_ebrw</th>
      <th>sat_math</th>
      <th>sat_total</th>
      <th>act_part_rate</th>
      <th>act_eng</th>
      <th>act_math</th>
      <th>act_reading</th>
      <th>act_sci</th>
      <th>act_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>100</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>38</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
      <td>65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>19.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>30</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>100</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>53</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
      <td>31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>22.8</td>
    </tr>
  </tbody>
</table>
</div>



### Save 2017 Merged data


```python
test_2017.to_csv('data/combined_2017.csv')
```

## Data Cleaning 2018 Dataset


```python
# import data
sat_2018 = pd.read_csv('data/sat_2018.csv')
act_2018 = pd.read_csv('data/act_2018_updated.csv')
```


```python
sat_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Participation</th>
      <th>Evidence-Based Reading and Writing</th>
      <th>Math</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>6%</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>43%</td>
      <td>562</td>
      <td>544</td>
      <td>1106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>29%</td>
      <td>577</td>
      <td>572</td>
      <td>1149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>5%</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>60%</td>
      <td>540</td>
      <td>536</td>
      <td>1076</td>
    </tr>
  </tbody>
</table>
</div>




```python
sat_2018.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51 entries, 0 to 50
    Data columns (total 5 columns):
     #   Column                              Non-Null Count  Dtype
    ---  ------                              --------------  -----
     0   State                               51 non-null     object
     1   Participation                       51 non-null     object
     2   Evidence-Based Reading and Writing  51 non-null     int64
     3   Math                                51 non-null     int64
     4   Total                               51 non-null     int64
    dtypes: int64(3), object(2)
    memory usage: 2.1+ KB



```python
act_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>Percentage of Students Tested</th>
      <th>Average Composite Score</th>
      <th>Average English Score</th>
      <th>Average Math Score</th>
      <th>Average Reading Score</th>
      <th>Average Science Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>33</td>
      <td>20.8</td>
      <td>19.8</td>
      <td>20.6</td>
      <td>21.6</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>66</td>
      <td>19.2</td>
      <td>18.2</td>
      <td>19.4</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>27</td>
      <td>22.7</td>
      <td>22.5</td>
      <td>22.5</td>
      <td>23.0</td>
      <td>22.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2018.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51 entries, 0 to 50
    Data columns (total 7 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   State                          51 non-null     object
     1   Percentage of Students Tested  51 non-null     int64  
     2   Average Composite Score        51 non-null     float64
     3   Average English Score          51 non-null     float64
     4   Average Math Score             51 non-null     float64
     5   Average Reading Score          51 non-null     float64
     6   Average Science Score          51 non-null     float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 2.9+ KB


### Change Participation Rate Data Type


```python
sat_2018['Participation'] = sat_2018['Participation'].apply(str.replace,args=('%',''))
sat_2018['Participation'] = sat_2018['Participation'].astype(int)
sat_2018.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51 entries, 0 to 50
    Data columns (total 5 columns):
     #   Column                              Non-Null Count  Dtype
    ---  ------                              --------------  -----
     0   State                               51 non-null     object
     1   Participation                       51 non-null     int32
     2   Evidence-Based Reading and Writing  51 non-null     int64
     3   Math                                51 non-null     int64
     4   Total                               51 non-null     int64
    dtypes: int32(1), int64(3), object(1)
    memory usage: 1.9+ KB


### Rename Columns


```python
header_dict.update({'percentage of students tested': 'part_rate',
                   'average composite score': 'total',
                   'average english score': 'eng',
                   'average math score': 'math',
                   'average reading score': 'reading',
                   'average science score': 'sci'})
```


```python
sat_2018.columns = sat_2018.columns.str.lower()
act_2018.columns = act_2018.columns.str.lower()
```


```python
sat_2018.rename(header_dict,axis=1, inplace=True)
sat_2018 = sat_2018.add_prefix('sat_')
sat_2018.rename({'sat_state':'state'}, axis=1 , inplace=True)

act_2018.rename(header_dict,axis=1, inplace=True)
act_2018 = act_2018.add_prefix('act_')
act_2018.rename({'act_state':'state'}, axis=1, inplace=True)
```


```python
sat_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate</th>
      <th>sat_ebrw</th>
      <th>sat_math</th>
      <th>sat_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>6</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>43</td>
      <td>562</td>
      <td>544</td>
      <td>1106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>29</td>
      <td>577</td>
      <td>572</td>
      <td>1149</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>5</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>60</td>
      <td>540</td>
      <td>536</td>
      <td>1076</td>
    </tr>
  </tbody>
</table>
</div>




```python
act_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>act_part_rate</th>
      <th>act_total</th>
      <th>act_eng</th>
      <th>act_math</th>
      <th>act_reading</th>
      <th>act_sci</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>33</td>
      <td>20.8</td>
      <td>19.8</td>
      <td>20.6</td>
      <td>21.6</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>66</td>
      <td>19.2</td>
      <td>18.2</td>
      <td>19.4</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>27</td>
      <td>22.7</td>
      <td>22.5</td>
      <td>22.5</td>
      <td>23.0</td>
      <td>22.1</td>
    </tr>
  </tbody>
</table>
</div>



### Merging of DataFrame


```python
# Print the shape of act_2017 & sat_2017
print(f'shape of act_2017 dataframe: {act_2018.shape}')
print(f'shape of sat_2017 dataframe: {sat_2018.shape}')
print('')
print('*' * 60)
print('')
# merge dataframe - common column is state
test_2018 = pd.merge(sat_2018, act_2018, how='outer', on='state')
print(f'shape of test_2017 dataframe: {test_2018.shape}')
```

    shape of act_2017 dataframe: (51, 7)
    shape of sat_2017 dataframe: (51, 5)

    ************************************************************

    shape of test_2017 dataframe: (51, 11)



```python
test_2018.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate</th>
      <th>sat_ebrw</th>
      <th>sat_math</th>
      <th>sat_total</th>
      <th>act_part_rate</th>
      <th>act_total</th>
      <th>act_eng</th>
      <th>act_math</th>
      <th>act_reading</th>
      <th>act_sci</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>6</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>43</td>
      <td>562</td>
      <td>544</td>
      <td>1106</td>
      <td>33</td>
      <td>20.8</td>
      <td>19.8</td>
      <td>20.6</td>
      <td>21.6</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>29</td>
      <td>577</td>
      <td>572</td>
      <td>1149</td>
      <td>66</td>
      <td>19.2</td>
      <td>18.2</td>
      <td>19.4</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>5</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>60</td>
      <td>540</td>
      <td>536</td>
      <td>1076</td>
      <td>27</td>
      <td>22.7</td>
      <td>22.5</td>
      <td>22.5</td>
      <td>23.0</td>
      <td>22.1</td>
    </tr>
  </tbody>
</table>
</div>



### Save 2018 Datasets


```python
test_2018.to_csv('data/combined_2018.csv')
```

### Combine 2 test datasets


```python
# Merge 2017 & 2018 data set
test_data = pd.merge(test_2017,test_2018, how='outer', on='state', suffixes=['_2017', '_2018'])

# Check Test dataset
print(f'Shape of test dataframe: {test_data.shape}')
test_data.head()
```

    Shape of test dataframe: (51, 21)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>100</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>...</td>
      <td>6</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>38</td>
      <td>547</td>
      <td>533</td>
      <td>1080</td>
      <td>65</td>
      <td>18.7</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>19.9</td>
      <td>...</td>
      <td>43</td>
      <td>562</td>
      <td>544</td>
      <td>1106</td>
      <td>33</td>
      <td>20.8</td>
      <td>19.8</td>
      <td>20.6</td>
      <td>21.6</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>30</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>62</td>
      <td>18.6</td>
      <td>19.8</td>
      <td>20.1</td>
      <td>19.8</td>
      <td>...</td>
      <td>29</td>
      <td>577</td>
      <td>572</td>
      <td>1149</td>
      <td>66</td>
      <td>19.2</td>
      <td>18.2</td>
      <td>19.4</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>100</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>...</td>
      <td>5</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>53</td>
      <td>531</td>
      <td>524</td>
      <td>1055</td>
      <td>31</td>
      <td>22.5</td>
      <td>22.7</td>
      <td>23.1</td>
      <td>22.2</td>
      <td>...</td>
      <td>60</td>
      <td>540</td>
      <td>536</td>
      <td>1076</td>
      <td>27</td>
      <td>22.7</td>
      <td>22.5</td>
      <td>22.5</td>
      <td>23.0</td>
      <td>22.1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Save combined data as a csv
test_data.to_csv('data/final.csv')
```

## Exploratory Data Analysis

### Summary Statistics


```python
test_data.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sat_part_rate_2017</th>
      <td>51.0</td>
      <td>39.803922</td>
      <td>35.276632</td>
      <td>2.0</td>
      <td>4.00</td>
      <td>38.0</td>
      <td>66.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>sat_ebrw_2017</th>
      <td>51.0</td>
      <td>569.117647</td>
      <td>45.666901</td>
      <td>482.0</td>
      <td>533.50</td>
      <td>559.0</td>
      <td>613.00</td>
      <td>644.0</td>
    </tr>
    <tr>
      <th>sat_math_2017</th>
      <td>51.0</td>
      <td>547.627451</td>
      <td>84.909119</td>
      <td>52.0</td>
      <td>522.00</td>
      <td>548.0</td>
      <td>599.00</td>
      <td>651.0</td>
    </tr>
    <tr>
      <th>sat_total_2017</th>
      <td>51.0</td>
      <td>1126.098039</td>
      <td>92.494812</td>
      <td>950.0</td>
      <td>1055.50</td>
      <td>1107.0</td>
      <td>1212.00</td>
      <td>1295.0</td>
    </tr>
    <tr>
      <th>act_part_rate_2017</th>
      <td>51.0</td>
      <td>65.254902</td>
      <td>32.140842</td>
      <td>8.0</td>
      <td>31.00</td>
      <td>69.0</td>
      <td>100.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>act_eng_2017</th>
      <td>51.0</td>
      <td>20.931373</td>
      <td>2.353677</td>
      <td>16.3</td>
      <td>19.00</td>
      <td>20.7</td>
      <td>23.30</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>act_math_2017</th>
      <td>51.0</td>
      <td>21.182353</td>
      <td>1.981989</td>
      <td>18.0</td>
      <td>19.40</td>
      <td>20.9</td>
      <td>23.10</td>
      <td>25.3</td>
    </tr>
    <tr>
      <th>act_reading_2017</th>
      <td>51.0</td>
      <td>22.013725</td>
      <td>2.067271</td>
      <td>18.1</td>
      <td>20.45</td>
      <td>21.8</td>
      <td>24.15</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>act_sci_2017</th>
      <td>51.0</td>
      <td>21.041176</td>
      <td>3.182463</td>
      <td>2.3</td>
      <td>19.90</td>
      <td>21.3</td>
      <td>22.75</td>
      <td>24.9</td>
    </tr>
    <tr>
      <th>act_total_2017</th>
      <td>51.0</td>
      <td>21.519608</td>
      <td>2.020695</td>
      <td>17.8</td>
      <td>19.80</td>
      <td>21.4</td>
      <td>23.60</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>sat_part_rate_2018</th>
      <td>51.0</td>
      <td>45.745098</td>
      <td>37.314256</td>
      <td>2.0</td>
      <td>4.50</td>
      <td>52.0</td>
      <td>77.50</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>sat_ebrw_2018</th>
      <td>51.0</td>
      <td>563.686275</td>
      <td>47.502627</td>
      <td>480.0</td>
      <td>534.50</td>
      <td>552.0</td>
      <td>610.50</td>
      <td>643.0</td>
    </tr>
    <tr>
      <th>sat_math_2018</th>
      <td>51.0</td>
      <td>556.235294</td>
      <td>47.772623</td>
      <td>480.0</td>
      <td>522.50</td>
      <td>544.0</td>
      <td>593.50</td>
      <td>655.0</td>
    </tr>
    <tr>
      <th>sat_total_2018</th>
      <td>51.0</td>
      <td>1120.019608</td>
      <td>94.155083</td>
      <td>977.0</td>
      <td>1057.50</td>
      <td>1098.0</td>
      <td>1204.00</td>
      <td>1298.0</td>
    </tr>
    <tr>
      <th>act_part_rate_2018</th>
      <td>51.0</td>
      <td>61.647059</td>
      <td>34.080976</td>
      <td>7.0</td>
      <td>28.50</td>
      <td>66.0</td>
      <td>100.00</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>act_total_2018</th>
      <td>51.0</td>
      <td>21.486275</td>
      <td>2.106278</td>
      <td>17.7</td>
      <td>19.95</td>
      <td>21.3</td>
      <td>23.55</td>
      <td>25.6</td>
    </tr>
    <tr>
      <th>act_eng_2018</th>
      <td>51.0</td>
      <td>20.988235</td>
      <td>2.446356</td>
      <td>16.6</td>
      <td>19.10</td>
      <td>20.2</td>
      <td>23.70</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>act_math_2018</th>
      <td>51.0</td>
      <td>21.125490</td>
      <td>2.035765</td>
      <td>17.8</td>
      <td>19.40</td>
      <td>20.7</td>
      <td>23.15</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>act_reading_2018</th>
      <td>51.0</td>
      <td>22.015686</td>
      <td>2.167245</td>
      <td>18.0</td>
      <td>20.45</td>
      <td>21.6</td>
      <td>24.10</td>
      <td>26.1</td>
    </tr>
    <tr>
      <th>act_sci_2018</th>
      <td>51.0</td>
      <td>21.345098</td>
      <td>1.870114</td>
      <td>17.9</td>
      <td>19.85</td>
      <td>21.1</td>
      <td>23.05</td>
      <td>24.9</td>
    </tr>
  </tbody>
</table>
</div>



### Manually calculate standard deviation

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^n(x_i - \mu)^2}$$

- Write a function to calculate standard deviation using the formula above


```python
def manual_std(data_series):
    return round(((((data_series - np.mean(data_series))**2).sum()) / len(data_series)) ** 0.5,5)
```


```python
# Create a dictionary comprehension to apply standard deviation function to each numeric column in the dataframe
{i:manual_std(test_data[i]) for i in test_data.select_dtypes(include=np.number).columns}
```




    {'sat_part_rate_2017': 34.92907,
     'sat_ebrw_2017': 45.21697,
     'sat_math_2017': 84.07256,
     'sat_total_2017': 91.58351,
     'act_part_rate_2017': 31.82418,
     'act_eng_2017': 2.33049,
     'act_math_2017': 1.96246,
     'act_reading_2017': 2.0469,
     'act_sci_2017': 3.15111,
     'act_total_2017': 2.00079,
     'sat_part_rate_2018': 36.94662,
     'sat_ebrw_2018': 47.03461,
     'sat_math_2018': 47.30195,
     'sat_total_2018': 93.22742,
     'act_part_rate_2018': 33.74519,
     'act_total_2018': 2.08553,
     'act_eng_2018': 2.42225,
     'act_math_2018': 2.01571,
     'act_reading_2018': 2.14589,
     'act_sci_2018': 1.85169}



Do your manually calculated standard deviations match up with the output from pandas `describe`? What about numpy's `std` method?


```python
# Test of function
print('std[sat_math] from .describe(): {}'.format(round(test_data['sat_math_2017'].describe()['std'],5)))
print('std[sat_math] from .manual_std():{}'.format(manual_std(test_data['sat_math_2017'])))
print('std[sat_math] from .np.std(): {}'.format(round(np.std(test_data['sat_math_2017']),5)))
```

    std[sat_math] from .describe(): 84.90912
    std[sat_math] from .manual_std():84.07256
    std[sat_math] from .np.std(): 84.07256


<span style = "color:Magenta">Analysis & Comments:</span><br>
Python.describe() uses an unbiased estimator (N-1) while Numpy & manual_std use N. Hence resulting in the difference.

For further understading refer to the links below:
<br><br>
<a href="https://stackoverflow.com/questions/24984178/different-std-in-pandas-vs-numpy">Explanation on standard devidation using Numpy versus Manual</a>
<br>
<a href='https://www.statisticshowto.com/bessels-correction/'>Bessels Correction</a>


### Which States have the highest & lowest participation rates in 2017 & 2018 SAT / ACT

Create a function to perform aggregation


```python
test_data.columns
def filter_mask(dataframe, one_column, one_aggregate):
    return dataframe[one_column] == dataframe[one_column].aggregate(one_aggregate)
```

#### SAT Max & Min Participation Rate


```python
# Find 2017 max & min participation rate
print(f'SAT_2017 Max Part Rate : {test_data["sat_part_rate_2017"].max()}')
print(f'SAT_2017 Min Part Rate : {test_data["sat_part_rate_2017"].min()}')
```

    SAT_2017 Max Part Rate : 100
    SAT_2017 Min Part Rate : 2



```python
# List of state with max participation rate
test_data[test_data['sat_part_rate_2017']==100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>100</td>
      <td>530</td>
      <td>512</td>
      <td>1041</td>
      <td>31</td>
      <td>25.5</td>
      <td>24.6</td>
      <td>25.6</td>
      <td>24.6</td>
      <td>...</td>
      <td>100</td>
      <td>535</td>
      <td>519</td>
      <td>1053</td>
      <td>26</td>
      <td>25.6</td>
      <td>26.0</td>
      <td>24.8</td>
      <td>26.1</td>
      <td>24.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>100</td>
      <td>503</td>
      <td>492</td>
      <td>996</td>
      <td>18</td>
      <td>24.1</td>
      <td>23.4</td>
      <td>24.8</td>
      <td>23.6</td>
      <td>...</td>
      <td>100</td>
      <td>505</td>
      <td>492</td>
      <td>998</td>
      <td>17</td>
      <td>23.2</td>
      <td>23.7</td>
      <td>23.1</td>
      <td>24.5</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>100</td>
      <td>482</td>
      <td>468</td>
      <td>950</td>
      <td>32</td>
      <td>24.4</td>
      <td>23.5</td>
      <td>24.9</td>
      <td>23.5</td>
      <td>...</td>
      <td>92</td>
      <td>497</td>
      <td>480</td>
      <td>977</td>
      <td>32</td>
      <td>23.6</td>
      <td>23.7</td>
      <td>22.7</td>
      <td>24.4</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>100</td>
      <td>509</td>
      <td>495</td>
      <td>1005</td>
      <td>29</td>
      <td>24.1</td>
      <td>23.7</td>
      <td>24.5</td>
      <td>23.8</td>
      <td>...</td>
      <td>100</td>
      <td>511</td>
      <td>499</td>
      <td>1011</td>
      <td>22</td>
      <td>24.4</td>
      <td>24.5</td>
      <td>23.9</td>
      <td>24.7</td>
      <td>23.9</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 21 columns</p>
</div>




```python
# List of state with min participation rate
test_data[test_data['sat_part_rate_2017']==2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>Iowa</td>
      <td>2</td>
      <td>641</td>
      <td>635</td>
      <td>1275</td>
      <td>67</td>
      <td>21.2</td>
      <td>21.3</td>
      <td>22.6</td>
      <td>22.1</td>
      <td>...</td>
      <td>3</td>
      <td>634</td>
      <td>631</td>
      <td>1265</td>
      <td>68</td>
      <td>21.8</td>
      <td>21.0</td>
      <td>21.2</td>
      <td>22.5</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>2</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
      <td>100</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>18.8</td>
      <td>...</td>
      <td>3</td>
      <td>630</td>
      <td>606</td>
      <td>1236</td>
      <td>100</td>
      <td>18.6</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.9</td>
      <td>18.6</td>
    </tr>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>2</td>
      <td>635</td>
      <td>621</td>
      <td>1256</td>
      <td>98</td>
      <td>19.0</td>
      <td>20.4</td>
      <td>20.5</td>
      <td>20.6</td>
      <td>...</td>
      <td>2</td>
      <td>640</td>
      <td>643</td>
      <td>1283</td>
      <td>98</td>
      <td>20.3</td>
      <td>19.1</td>
      <td>20.3</td>
      <td>20.7</td>
      <td>20.5</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>




```python
# Find 2018 max & min participation rate
print(f'SAT_2018 Max Part Rate : {test_data["sat_part_rate_2018"].max()}')
print(f'SAT_2018 Min Part Rate : {test_data["sat_part_rate_2018"].min()}')
```

    SAT_2018 Max Part Rate : 100
    SAT_2018 Min Part Rate : 2



```python
# List of state with max participation rate
test_data[test_data['sat_part_rate_2018']==100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>11</td>
      <td>606</td>
      <td>595</td>
      <td>1201</td>
      <td>100</td>
      <td>20.1</td>
      <td>20.3</td>
      <td>21.2</td>
      <td>20.9</td>
      <td>...</td>
      <td>100</td>
      <td>519</td>
      <td>506</td>
      <td>1025</td>
      <td>30</td>
      <td>23.9</td>
      <td>23.9</td>
      <td>23.2</td>
      <td>24.4</td>
      <td>23.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>100</td>
      <td>530</td>
      <td>512</td>
      <td>1041</td>
      <td>31</td>
      <td>25.5</td>
      <td>24.6</td>
      <td>25.6</td>
      <td>24.6</td>
      <td>...</td>
      <td>100</td>
      <td>535</td>
      <td>519</td>
      <td>1053</td>
      <td>26</td>
      <td>25.6</td>
      <td>26.0</td>
      <td>24.8</td>
      <td>26.1</td>
      <td>24.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Delaware</td>
      <td>100</td>
      <td>503</td>
      <td>492</td>
      <td>996</td>
      <td>18</td>
      <td>24.1</td>
      <td>23.4</td>
      <td>24.8</td>
      <td>23.6</td>
      <td>...</td>
      <td>100</td>
      <td>505</td>
      <td>492</td>
      <td>998</td>
      <td>17</td>
      <td>23.2</td>
      <td>23.7</td>
      <td>23.1</td>
      <td>24.5</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>93</td>
      <td>513</td>
      <td>493</td>
      <td>1005</td>
      <td>38</td>
      <td>21.9</td>
      <td>21.8</td>
      <td>23.0</td>
      <td>22.1</td>
      <td>...</td>
      <td>100</td>
      <td>508</td>
      <td>493</td>
      <td>1001</td>
      <td>36</td>
      <td>22.3</td>
      <td>21.9</td>
      <td>21.6</td>
      <td>23.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Michigan</td>
      <td>100</td>
      <td>509</td>
      <td>495</td>
      <td>1005</td>
      <td>29</td>
      <td>24.1</td>
      <td>23.7</td>
      <td>24.5</td>
      <td>23.8</td>
      <td>...</td>
      <td>100</td>
      <td>511</td>
      <td>499</td>
      <td>1011</td>
      <td>22</td>
      <td>24.4</td>
      <td>24.5</td>
      <td>23.9</td>
      <td>24.7</td>
      <td>23.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# List of state with min participation rate
test_data[test_data['sat_part_rate_2018']==2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>North Dakota</td>
      <td>2</td>
      <td>635</td>
      <td>621</td>
      <td>1256</td>
      <td>98</td>
      <td>19.0</td>
      <td>20.4</td>
      <td>20.5</td>
      <td>20.6</td>
      <td>...</td>
      <td>2</td>
      <td>640</td>
      <td>643</td>
      <td>1283</td>
      <td>98</td>
      <td>20.3</td>
      <td>19.1</td>
      <td>20.3</td>
      <td>20.7</td>
      <td>20.5</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



#### ACT Max & Min Participation Rate


```python
# Find 2017 max & min participation rate
print(f'ACT_2017 Max Part Rate : {test_data["act_part_rate_2017"].max()}')
print(f'ACT_2017 Min Part Rate : {test_data["act_part_rate_2017"].min()}')
```

    ACT_2017 Max Part Rate : 100
    ACT_2017 Min Part Rate : 8



```python
# List of state with max participation rate
test_data[test_data['act_part_rate_2017']==100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>100</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>...</td>
      <td>6</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>100</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>...</td>
      <td>5</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>11</td>
      <td>606</td>
      <td>595</td>
      <td>1201</td>
      <td>100</td>
      <td>20.1</td>
      <td>20.3</td>
      <td>21.2</td>
      <td>20.9</td>
      <td>...</td>
      <td>100</td>
      <td>519</td>
      <td>506</td>
      <td>1025</td>
      <td>30</td>
      <td>23.9</td>
      <td>23.9</td>
      <td>23.2</td>
      <td>24.4</td>
      <td>23.5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>4</td>
      <td>631</td>
      <td>616</td>
      <td>1247</td>
      <td>100</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>20.5</td>
      <td>20.1</td>
      <td>...</td>
      <td>4</td>
      <td>630</td>
      <td>618</td>
      <td>1248</td>
      <td>100</td>
      <td>20.2</td>
      <td>19.9</td>
      <td>19.7</td>
      <td>20.8</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>4</td>
      <td>611</td>
      <td>586</td>
      <td>1198</td>
      <td>100</td>
      <td>19.4</td>
      <td>18.8</td>
      <td>19.8</td>
      <td>19.6</td>
      <td>...</td>
      <td>4</td>
      <td>615</td>
      <td>595</td>
      <td>1210</td>
      <td>100</td>
      <td>19.2</td>
      <td>19.0</td>
      <td>18.5</td>
      <td>19.6</td>
      <td>19.1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>3</td>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>100</td>
      <td>20.4</td>
      <td>21.5</td>
      <td>21.8</td>
      <td>21.6</td>
      <td>...</td>
      <td>4</td>
      <td>643</td>
      <td>655</td>
      <td>1298</td>
      <td>99</td>
      <td>21.3</td>
      <td>20.2</td>
      <td>21.4</td>
      <td>21.7</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>2</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
      <td>100</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>18.8</td>
      <td>...</td>
      <td>3</td>
      <td>630</td>
      <td>606</td>
      <td>1236</td>
      <td>100</td>
      <td>18.6</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.9</td>
      <td>18.6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>3</td>
      <td>640</td>
      <td>631</td>
      <td>1271</td>
      <td>100</td>
      <td>19.8</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.5</td>
      <td>...</td>
      <td>4</td>
      <td>633</td>
      <td>629</td>
      <td>1262</td>
      <td>100</td>
      <td>20.0</td>
      <td>19.5</td>
      <td>19.7</td>
      <td>20.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>10</td>
      <td>605</td>
      <td>591</td>
      <td>1196</td>
      <td>100</td>
      <td>19.0</td>
      <td>20.2</td>
      <td>21.0</td>
      <td>20.5</td>
      <td>...</td>
      <td>10</td>
      <td>606</td>
      <td>592</td>
      <td>1198</td>
      <td>100</td>
      <td>20.0</td>
      <td>18.7</td>
      <td>19.9</td>
      <td>20.7</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>26</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>100</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>18.2</td>
      <td>...</td>
      <td>23</td>
      <td>574</td>
      <td>566</td>
      <td>1140</td>
      <td>100</td>
      <td>17.7</td>
      <td>16.6</td>
      <td>17.8</td>
      <td>18.0</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>49</td>
      <td>546</td>
      <td>535</td>
      <td>1081</td>
      <td>100</td>
      <td>17.8</td>
      <td>19.3</td>
      <td>19.6</td>
      <td>19.3</td>
      <td>...</td>
      <td>52</td>
      <td>554</td>
      <td>543</td>
      <td>1098</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.0</td>
      <td>19.3</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>7</td>
      <td>530</td>
      <td>517</td>
      <td>1047</td>
      <td>100</td>
      <td>18.5</td>
      <td>18.8</td>
      <td>20.1</td>
      <td>19.6</td>
      <td>...</td>
      <td>8</td>
      <td>541</td>
      <td>521</td>
      <td>1062</td>
      <td>100</td>
      <td>19.3</td>
      <td>18.4</td>
      <td>18.8</td>
      <td>20.1</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Carolina</td>
      <td>50</td>
      <td>543</td>
      <td>521</td>
      <td>1064</td>
      <td>100</td>
      <td>17.5</td>
      <td>18.6</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>...</td>
      <td>55</td>
      <td>547</td>
      <td>523</td>
      <td>1070</td>
      <td>100</td>
      <td>18.3</td>
      <td>17.3</td>
      <td>18.2</td>
      <td>18.6</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tennessee</td>
      <td>5</td>
      <td>623</td>
      <td>604</td>
      <td>1228</td>
      <td>100</td>
      <td>19.5</td>
      <td>19.2</td>
      <td>20.1</td>
      <td>19.9</td>
      <td>...</td>
      <td>6</td>
      <td>624</td>
      <td>607</td>
      <td>1231</td>
      <td>100</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>19.9</td>
      <td>19.6</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Utah</td>
      <td>3</td>
      <td>624</td>
      <td>614</td>
      <td>1238</td>
      <td>100</td>
      <td>19.5</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>...</td>
      <td>4</td>
      <td>480</td>
      <td>530</td>
      <td>1010</td>
      <td>100</td>
      <td>20.4</td>
      <td>19.7</td>
      <td>19.9</td>
      <td>20.9</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wisconsin</td>
      <td>3</td>
      <td>642</td>
      <td>649</td>
      <td>1291</td>
      <td>100</td>
      <td>19.7</td>
      <td>20.4</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>...</td>
      <td>3</td>
      <td>641</td>
      <td>653</td>
      <td>1294</td>
      <td>100</td>
      <td>20.5</td>
      <td>19.8</td>
      <td>20.3</td>
      <td>20.6</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Wyoming</td>
      <td>3</td>
      <td>626</td>
      <td>604</td>
      <td>1230</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.8</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>...</td>
      <td>3</td>
      <td>633</td>
      <td>625</td>
      <td>1257</td>
      <td>100</td>
      <td>20.0</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 21 columns</p>
</div>




```python
# List of state with min participation rate
test_data[test_data['act_part_rate_2017']==8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>95</td>
      <td>513</td>
      <td>499</td>
      <td>1012</td>
      <td>8</td>
      <td>24.2</td>
      <td>24.0</td>
      <td>24.8</td>
      <td>23.7</td>
      <td>...</td>
      <td>99</td>
      <td>512</td>
      <td>501</td>
      <td>1013</td>
      <td>7</td>
      <td>24.0</td>
      <td>23.8</td>
      <td>23.6</td>
      <td>24.7</td>
      <td>23.4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# Find 2018 max & min participation rate
print(f'ACT_2018 Max Part Rate : {test_data["act_part_rate_2018"].max()}')
print(f'ACT_2018 Min Part Rate : {test_data["act_part_rate_2018"].min()}')
```

    ACT_2018 Max Part Rate : 100
    ACT_2018 Min Part Rate : 7



```python
# List of state with max participation rate
test_data[test_data['act_part_rate_2018']==100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>5</td>
      <td>593</td>
      <td>572</td>
      <td>1165</td>
      <td>100</td>
      <td>18.9</td>
      <td>18.4</td>
      <td>19.7</td>
      <td>19.4</td>
      <td>...</td>
      <td>6</td>
      <td>595</td>
      <td>571</td>
      <td>1166</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>18.3</td>
      <td>19.6</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>3</td>
      <td>614</td>
      <td>594</td>
      <td>1208</td>
      <td>100</td>
      <td>18.9</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>19.5</td>
      <td>...</td>
      <td>5</td>
      <td>592</td>
      <td>576</td>
      <td>1169</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>19.7</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kentucky</td>
      <td>4</td>
      <td>631</td>
      <td>616</td>
      <td>1247</td>
      <td>100</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>20.5</td>
      <td>20.1</td>
      <td>...</td>
      <td>4</td>
      <td>630</td>
      <td>618</td>
      <td>1248</td>
      <td>100</td>
      <td>20.2</td>
      <td>19.9</td>
      <td>19.7</td>
      <td>20.8</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Louisiana</td>
      <td>4</td>
      <td>611</td>
      <td>586</td>
      <td>1198</td>
      <td>100</td>
      <td>19.4</td>
      <td>18.8</td>
      <td>19.8</td>
      <td>19.6</td>
      <td>...</td>
      <td>4</td>
      <td>615</td>
      <td>595</td>
      <td>1210</td>
      <td>100</td>
      <td>19.2</td>
      <td>19.0</td>
      <td>18.5</td>
      <td>19.6</td>
      <td>19.1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Mississippi</td>
      <td>2</td>
      <td>634</td>
      <td>607</td>
      <td>1242</td>
      <td>100</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.8</td>
      <td>18.8</td>
      <td>...</td>
      <td>3</td>
      <td>630</td>
      <td>606</td>
      <td>1236</td>
      <td>100</td>
      <td>18.6</td>
      <td>18.2</td>
      <td>18.1</td>
      <td>18.9</td>
      <td>18.6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Missouri</td>
      <td>3</td>
      <td>640</td>
      <td>631</td>
      <td>1271</td>
      <td>100</td>
      <td>19.8</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.5</td>
      <td>...</td>
      <td>4</td>
      <td>633</td>
      <td>629</td>
      <td>1262</td>
      <td>100</td>
      <td>20.0</td>
      <td>19.5</td>
      <td>19.7</td>
      <td>20.5</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Montana</td>
      <td>10</td>
      <td>605</td>
      <td>591</td>
      <td>1196</td>
      <td>100</td>
      <td>19.0</td>
      <td>20.2</td>
      <td>21.0</td>
      <td>20.5</td>
      <td>...</td>
      <td>10</td>
      <td>606</td>
      <td>592</td>
      <td>1198</td>
      <td>100</td>
      <td>20.0</td>
      <td>18.7</td>
      <td>19.9</td>
      <td>20.7</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>3</td>
      <td>629</td>
      <td>625</td>
      <td>1253</td>
      <td>84</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.9</td>
      <td>21.5</td>
      <td>...</td>
      <td>3</td>
      <td>629</td>
      <td>623</td>
      <td>1252</td>
      <td>100</td>
      <td>20.1</td>
      <td>19.4</td>
      <td>19.8</td>
      <td>20.4</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>26</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>100</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>18.2</td>
      <td>...</td>
      <td>23</td>
      <td>574</td>
      <td>566</td>
      <td>1140</td>
      <td>100</td>
      <td>17.7</td>
      <td>16.6</td>
      <td>17.8</td>
      <td>18.0</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>49</td>
      <td>546</td>
      <td>535</td>
      <td>1081</td>
      <td>100</td>
      <td>17.8</td>
      <td>19.3</td>
      <td>19.6</td>
      <td>19.3</td>
      <td>...</td>
      <td>52</td>
      <td>554</td>
      <td>543</td>
      <td>1098</td>
      <td>100</td>
      <td>19.1</td>
      <td>18.0</td>
      <td>19.3</td>
      <td>19.5</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>12</td>
      <td>578</td>
      <td>570</td>
      <td>1149</td>
      <td>75</td>
      <td>21.2</td>
      <td>21.6</td>
      <td>22.5</td>
      <td>22.0</td>
      <td>...</td>
      <td>18</td>
      <td>552</td>
      <td>547</td>
      <td>1099</td>
      <td>100</td>
      <td>20.3</td>
      <td>19.3</td>
      <td>20.3</td>
      <td>20.8</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Oklahoma</td>
      <td>7</td>
      <td>530</td>
      <td>517</td>
      <td>1047</td>
      <td>100</td>
      <td>18.5</td>
      <td>18.8</td>
      <td>20.1</td>
      <td>19.6</td>
      <td>...</td>
      <td>8</td>
      <td>541</td>
      <td>521</td>
      <td>1062</td>
      <td>100</td>
      <td>19.3</td>
      <td>18.4</td>
      <td>18.8</td>
      <td>20.1</td>
      <td>19.4</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Carolina</td>
      <td>50</td>
      <td>543</td>
      <td>521</td>
      <td>1064</td>
      <td>100</td>
      <td>17.5</td>
      <td>18.6</td>
      <td>19.1</td>
      <td>18.9</td>
      <td>...</td>
      <td>55</td>
      <td>547</td>
      <td>523</td>
      <td>1070</td>
      <td>100</td>
      <td>18.3</td>
      <td>17.3</td>
      <td>18.2</td>
      <td>18.6</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Tennessee</td>
      <td>5</td>
      <td>623</td>
      <td>604</td>
      <td>1228</td>
      <td>100</td>
      <td>19.5</td>
      <td>19.2</td>
      <td>20.1</td>
      <td>19.9</td>
      <td>...</td>
      <td>6</td>
      <td>624</td>
      <td>607</td>
      <td>1231</td>
      <td>100</td>
      <td>19.6</td>
      <td>19.4</td>
      <td>19.1</td>
      <td>19.9</td>
      <td>19.6</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Utah</td>
      <td>3</td>
      <td>624</td>
      <td>614</td>
      <td>1238</td>
      <td>100</td>
      <td>19.5</td>
      <td>19.9</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>...</td>
      <td>4</td>
      <td>480</td>
      <td>530</td>
      <td>1010</td>
      <td>100</td>
      <td>20.4</td>
      <td>19.7</td>
      <td>19.9</td>
      <td>20.9</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Wisconsin</td>
      <td>3</td>
      <td>642</td>
      <td>649</td>
      <td>1291</td>
      <td>100</td>
      <td>19.7</td>
      <td>20.4</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>...</td>
      <td>3</td>
      <td>641</td>
      <td>653</td>
      <td>1294</td>
      <td>100</td>
      <td>20.5</td>
      <td>19.8</td>
      <td>20.3</td>
      <td>20.6</td>
      <td>20.8</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Wyoming</td>
      <td>3</td>
      <td>626</td>
      <td>604</td>
      <td>1230</td>
      <td>100</td>
      <td>19.4</td>
      <td>19.8</td>
      <td>20.8</td>
      <td>20.6</td>
      <td>...</td>
      <td>3</td>
      <td>633</td>
      <td>625</td>
      <td>1257</td>
      <td>100</td>
      <td>20.0</td>
      <td>19.0</td>
      <td>19.7</td>
      <td>20.6</td>
      <td>20.3</td>
    </tr>
  </tbody>
</table>
<p>17 rows × 21 columns</p>
</div>




```python
# List of state with min participation rate
test_data[test_data['act_part_rate_2018']==7]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>Maine</td>
      <td>95</td>
      <td>513</td>
      <td>499</td>
      <td>1012</td>
      <td>8</td>
      <td>24.2</td>
      <td>24.0</td>
      <td>24.8</td>
      <td>23.7</td>
      <td>...</td>
      <td>99</td>
      <td>512</td>
      <td>501</td>
      <td>1013</td>
      <td>7</td>
      <td>24.0</td>
      <td>23.8</td>
      <td>23.6</td>
      <td>24.7</td>
      <td>23.4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



### Which states have the highest and lowest mean total/composite scores for 2017 SAT | 2018 SAT | 2017 ACT | 2018 ACT

#### SAT Max & Min Total Score


```python
# Find 2017 max & min total score
print(f'SAT_2017 Max Total Score : {test_data["sat_total_2017"].max()}')
print(f'SAT_2017 Min Total Score : {test_data["sat_total_2017"].min()}')
```

    SAT_2017 Max Total Score : 1295
    SAT_2017 Min Total Score : 950



```python
# List of state with max total score
test_data[test_data['sat_total_2017']==1295]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>3</td>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>100</td>
      <td>20.4</td>
      <td>21.5</td>
      <td>21.8</td>
      <td>21.6</td>
      <td>...</td>
      <td>4</td>
      <td>643</td>
      <td>655</td>
      <td>1298</td>
      <td>99</td>
      <td>21.3</td>
      <td>20.2</td>
      <td>21.4</td>
      <td>21.7</td>
      <td>21.4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# List of state with min total score
test_data[test_data['sat_total_2017']==950]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>100</td>
      <td>482</td>
      <td>468</td>
      <td>950</td>
      <td>32</td>
      <td>24.4</td>
      <td>23.5</td>
      <td>24.9</td>
      <td>23.5</td>
      <td>...</td>
      <td>92</td>
      <td>497</td>
      <td>480</td>
      <td>977</td>
      <td>32</td>
      <td>23.6</td>
      <td>23.7</td>
      <td>22.7</td>
      <td>24.4</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# Find 2018 max & min total score
print(f'SAT_2018 Max Total Score : {test_data["sat_total_2018"].max()}')
print(f'SAT_2018 Min Total Score : {test_data["sat_total_2018"].min()}')
```

    SAT_2018 Max Total Score : 1298
    SAT_2018 Min Total Score : 977



```python
# List of state with max total score
test_data[test_data['sat_total_2018']==1298]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>3</td>
      <td>644</td>
      <td>651</td>
      <td>1295</td>
      <td>100</td>
      <td>20.4</td>
      <td>21.5</td>
      <td>21.8</td>
      <td>21.6</td>
      <td>...</td>
      <td>4</td>
      <td>643</td>
      <td>655</td>
      <td>1298</td>
      <td>99</td>
      <td>21.3</td>
      <td>20.2</td>
      <td>21.4</td>
      <td>21.7</td>
      <td>21.4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# List of state with min total score
test_data[test_data['sat_total_2018']==977]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>100</td>
      <td>482</td>
      <td>468</td>
      <td>950</td>
      <td>32</td>
      <td>24.4</td>
      <td>23.5</td>
      <td>24.9</td>
      <td>23.5</td>
      <td>...</td>
      <td>92</td>
      <td>497</td>
      <td>480</td>
      <td>977</td>
      <td>32</td>
      <td>23.6</td>
      <td>23.7</td>
      <td>22.7</td>
      <td>24.4</td>
      <td>23.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



#### ACT Max & Min Total Score


```python
# Find 2017 max & min total score
print(f'ACT_2017 Max Total Score : {test_data["act_total_2017"].max()}')
print(f'ACT_2017 Min Total Score : {test_data["act_total_2017"].min()}')
```

    ACT_2017 Max Total Score : 25.5
    ACT_2017 Min Total Score : 17.8



```python
# List of state with max total score
test_data[test_data['act_total_2017']==25.5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>New Hampshire</td>
      <td>96</td>
      <td>532</td>
      <td>520</td>
      <td>1052</td>
      <td>18</td>
      <td>25.4</td>
      <td>25.1</td>
      <td>26.0</td>
      <td>24.9</td>
      <td>...</td>
      <td>96</td>
      <td>535</td>
      <td>528</td>
      <td>1063</td>
      <td>16</td>
      <td>25.1</td>
      <td>25.1</td>
      <td>24.7</td>
      <td>25.6</td>
      <td>24.4</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# List of state with min total score
test_data[test_data['act_total_2017']==17.8]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>26</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>100</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>18.2</td>
      <td>...</td>
      <td>23</td>
      <td>574</td>
      <td>566</td>
      <td>1140</td>
      <td>100</td>
      <td>17.7</td>
      <td>16.6</td>
      <td>17.8</td>
      <td>18.0</td>
      <td>17.9</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# Find 2018 max & min total score
print(f'ACT_2018 Max Total Score : {test_data["act_total_2018"].max()}')
print(f'ACT_2018 Min Total Score : {test_data["act_total_2018"].min()}')
```

    ACT_2018 Max Total Score : 25.6
    ACT_2018 Min Total Score : 17.7



```python
# List of state with max total score
test_data[test_data['act_total_2018']==25.6]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Connecticut</td>
      <td>100</td>
      <td>530</td>
      <td>512</td>
      <td>1041</td>
      <td>31</td>
      <td>25.5</td>
      <td>24.6</td>
      <td>25.6</td>
      <td>24.6</td>
      <td>...</td>
      <td>100</td>
      <td>535</td>
      <td>519</td>
      <td>1053</td>
      <td>26</td>
      <td>25.6</td>
      <td>26.0</td>
      <td>24.8</td>
      <td>26.1</td>
      <td>24.9</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# List of state with min total score
test_data[test_data['act_total_2018']==17.7]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_ebrw_2017</th>
      <th>sat_math_2017</th>
      <th>sat_total_2017</th>
      <th>act_part_rate_2017</th>
      <th>act_eng_2017</th>
      <th>act_math_2017</th>
      <th>act_reading_2017</th>
      <th>act_sci_2017</th>
      <th>...</th>
      <th>sat_part_rate_2018</th>
      <th>sat_ebrw_2018</th>
      <th>sat_math_2018</th>
      <th>sat_total_2018</th>
      <th>act_part_rate_2018</th>
      <th>act_total_2018</th>
      <th>act_eng_2018</th>
      <th>act_math_2018</th>
      <th>act_reading_2018</th>
      <th>act_sci_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>Nevada</td>
      <td>26</td>
      <td>563</td>
      <td>553</td>
      <td>1116</td>
      <td>100</td>
      <td>16.3</td>
      <td>18.0</td>
      <td>18.1</td>
      <td>18.2</td>
      <td>...</td>
      <td>23</td>
      <td>574</td>
      <td>566</td>
      <td>1140</td>
      <td>100</td>
      <td>17.7</td>
      <td>16.6</td>
      <td>17.8</td>
      <td>18.0</td>
      <td>17.9</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



### Do any states with 100% participation on a given test have a rate change year-to-year?


```python
# Based on SAT participation rate in 2017
test_data[['state','sat_part_rate_2017','sat_part_rate_2018']][(test_data['sat_part_rate_2017']==100) & (test_data['sat_part_rate_2018']!=100) ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_part_rate_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>District of Columbia</td>
      <td>100</td>
      <td>92</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Based on SAT participation rate in 2018
test_data[['state','sat_part_rate_2017','sat_part_rate_2018']][(test_data['sat_part_rate_2017']!=100) & (test_data['sat_part_rate_2018']==100)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>sat_part_rate_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>11</td>
      <td>100</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Idaho</td>
      <td>93</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Based on ACT participation rate in 2017
test_data[['state','act_part_rate_2017','act_part_rate_2018']][(test_data['act_part_rate_2017']==100) & (test_data['act_part_rate_2018']!=100)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>act_part_rate_2017</th>
      <th>act_part_rate_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Colorado</td>
      <td>100</td>
      <td>30</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Minnesota</td>
      <td>100</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Based on ACT participation rate in 2018
test_data[['state','act_part_rate_2017','act_part_rate_2018']][(test_data['act_part_rate_2017']!=100) & (test_data['act_part_rate_2018']==100)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>act_part_rate_2017</th>
      <th>act_part_rate_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27</th>
      <td>Nebraska</td>
      <td>84</td>
      <td>100</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Ohio</td>
      <td>75</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



### Do any states show have >50% participation on both tests either year?


```python
# Year 2017
test_data[['state','sat_part_rate_2017','act_part_rate_2017']][(test_data['sat_part_rate_2017']>50) & (test_data['act_part_rate_2017']>50) ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2017</th>
      <th>act_part_rate_2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>83</td>
      <td>73</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>61</td>
      <td>55</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>55</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Year 2018
test_data[['state','sat_part_rate_2018','act_part_rate_2018']][(test_data['sat_part_rate_2018']>50) & (test_data['act_part_rate_2018']>50)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>sat_part_rate_2018</th>
      <th>act_part_rate_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Florida</td>
      <td>56</td>
      <td>66</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Georgia</td>
      <td>70</td>
      <td>53</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hawaii</td>
      <td>56</td>
      <td>89</td>
    </tr>
    <tr>
      <th>33</th>
      <td>North Carolina</td>
      <td>52</td>
      <td>100</td>
    </tr>
    <tr>
      <th>40</th>
      <td>South Carolina</td>
      <td>55</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>

## Conclusion & Recommendations

The participation rate in both test for each year (see scatterplot) reflect similar trend with states either preferring one over the other. Having said that, there seems to be no obvious preferrence of one over the other and changes in participation can likely result due to change in state's legislation or change of testing contractors.

ACT and SAT scores are inversely correlated with their respective participation rates. In non-mandatory states, participants only parttake the test if they believe they are prepared and able to achieve a high score in the test. While in states where exam are mandatory, participants may be 'forced' to take the exam regardless if they have the abilit y to beat the average.

SAT was more likely used by students living in coastal states and ACT was more widely used by students in the Midwest & South. Although there seem to be a shift in trend but the result as is could be due to regional and/or political affiliations associated with vendor of ACT or SAT.

SAT has seens increasing participation year-on-year as compared to ACT, which remains relative stable. Compared with ACT, SAT have gain more participation in 2018. This could be due to their initiatives to:
1. Offer free SAT 'school day' event, where participants can take the exam during regular school day with the cost covered by their schools, not their parents.
2. Partnering with other education providers to provide review lesson and practice tests.

Based on analysis of the data & additional research, I choose Florida as the next state we can invest in. The reason for choosing the state are as follow:

1. Florida is the 3rd largest state in term of population which means that there are likely to have more students participating in college admission test.

2. As an investment risk mitigation option, we may not be able to replicate the success case of Illinois & Colorado, low SAT to high SAT participation rate. Using Florida, which already have close to 50% participation rate in both SAT and ACT, it is likely to have less 'barrier to entry' for students to switch from ACT to SAT since they may be familiar with the structure of both exam.

3. Florida is committed to make big investment into education, hence it is highly likely to win political favor in expanding into Florida, given that our part of our current initiatives (free online review/classes) can contribute to more students being admitted to college.


College Board should continue with these initiatives for states which have a participation rate of 50% and above in 2018 going forward will most likely to see an increase in their popularity among the students.

Supporting:
    <ul>
        <li><a herf="https://www.collegeraptor.com/getting-in/articles/act-sat/preference-act-sat-state-infographic/"> State Exam Preference</a></li>
        <li><a herf="https://archive.nytimes.com/www.nytimes.com/interactive/2013/08/04/education/edlife/where-the-sat-and-act-dominate.html">where SAT & ACT dominate</a></li>
        <li><a herf="https://en.wikipedia.org/wiki/SAT"> SAT Prefernce</a></li>
        <li><a herf="https://www.orlandosentinel.com/news/education/os-ne-act-sat-florida-scores-20181024-story.html">SAT Initiative in Florida</a></li>
        <li><a herf="https://www.politifact.com/factchecks/2017/may/15/richard-corcoran/corcoran-touts-level-education-spending-there/"> Florida make big investment in education</a></li>
    </ul>


```python

```
