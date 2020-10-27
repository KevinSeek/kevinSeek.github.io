---
layout: post
title:  "Project Evian"
date:   2020-10-27 10:00:00 +0800
categories: ['Analytics', 'Web-Scraping', 'Natural Language Processing', 'Word Cloud', 'Classification Techniques']
excerpt_separator: <!--more-->
---


## Introduction
This is a project done as part of Data Science Immersive Course conducted by General Assembly. This project is to showcase the following skillsets:
- Conceptualization thinking
- Extraction, Transformation & Loading in python
- Web Scraping
- Natural Language Processing
- Classification Modelling

## CodeBook
The full codebook can be found [here](https://github.com/KevinSeek/Kevin_Project_Portfolio/tree/master/P003_project_evian)
<!--more-->

## Scenario

Seeking Alpha site is trying to understand the American consumer's interests in investing and to find the latest buzz words to provide more targeted portfolio offers to them. However, they realized that consumers/users to their sites have mixed interested in pure investment-related posts to anything and everything related to personal finance. This makes it hard for them to identify new customer bases of which they can offer their services too. Due to this problem, they have hired ML company to help them to address this issue.  

Upon taking up this project, ML company have assigned me, a junior data analyst, to help Seeking Alpha to solve this problem.

## Problem Statement

- Help-Seeking Alpha to build a classification model that help to classify their users' posts based on the words they use and where possible, help to identify consumer's lifestyle habits eg. Buzz words, the time where users are most active in generating posts related to investment.


- The model must be simple enough for non-technical executives to understand and must have at least an 80% success rate of correctly classifying the posts.  
    - The client, Seeking Alpha, understand that in the eyes of consumers, personal finance and investing are closely related and they accept and tolerate some misclassification errors - posts that are wrongly tagged as either personal finance or investing. 

## Executive Summary

Seeking Alpha is looking for a tool/model which can help them to accurately (~80%) identify users, based on their posts, to market to them their financial (investment) products. This marketing campaign is targeted to American netizen only and hence a suitable social news aggregator website, Reddit, is used to gather data. 2 subreddits - Investing and personal finance was chosen to train the model for the following reasons:  

1. Topics are similar; this can be a test to see if the model is robust enough to identify investing posts correctly against the contrasting topic - personal finance


2. Reduced loss - There is a minimum loss to the client if they target their financial products to the wrong groups, personal finance given the similar concerns the users might be facing; they may even convert them as buyers. This may not be the case if the contrasting topic is eg. baking. Users from that groups might complain against them for spam marketing.

**Key Observations**  
- Although selection criteria is that text-based posts where posts are longer than 255 characters are used. There is a sizable number of posts gathered suggesting that there are many users who are concerns about investing and/or personal finance - good business opportunity.


- Users are generally more active from Monday to Thursday, from 9 pm to midnight. This information may help the client to strategize their marketing efforts to ensure maximum reach.


- 2 text-processing techniques are used
    - Counter Vectorizer which simply ranks the importance of words based on the occurrence frequency
    - TF-IDF is a score that tells us which words are important to one document, relative to all other documents. Words that occur often in one document but don't occur in many documents contain more predictive power.


- 2 type of model are chosen: 
    - logistic regression classifier
    - Naive Bayes Model are trained & test.  


All four models' performance exceeds the minimum accuracy of 80% agreed with clients. TF-IDF Text processing technique and Naive Bayes Modeling were recommended based on business concerns.


## Extraction & Transforming Users' Posts

### Import data


```python
df = pd.read_csv('datasets/combined.csv')
df.head(4)
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
      <th>no_follow</th>
      <th>saved</th>
      <th>all_awardings</th>
      <th>top_awarded_type</th>
      <th>link_flair_background_color</th>
      <th>author_premium</th>
      <th>visited</th>
      <th>link_flair_type</th>
      <th>can_gild</th>
      <th>subreddit</th>
      <th>subreddit_id</th>
      <th>edited</th>
      <th>is_reddit_media_domain</th>
      <th>parent_whitelist_status</th>
      <th>mod_note</th>
      <th>approved_by</th>
      <th>hidden</th>
      <th>link_flair_richtext</th>
      <th>can_mod_post</th>
      <th>id</th>
      <th>author</th>
      <th>is_original_content</th>
      <th>subreddit_type</th>
      <th>content_categories</th>
      <th>locked</th>
      <th>author_fullname</th>
      <th>title</th>
      <th>hide_score</th>
      <th>gildings</th>
      <th>view_count</th>
      <th>author_cakeday</th>
      <th>likes</th>
      <th>url</th>
      <th>mod_reports</th>
      <th>author_flair_text_color</th>
      <th>gilded</th>
      <th>banned_at_utc</th>
      <th>is_self</th>
      <th>author_flair_template_id</th>
      <th>thumbnail</th>
      <th>suggested_sort</th>
      <th>media_only</th>
      <th>banned_by</th>
      <th>is_crosspostable</th>
      <th>discussion_type</th>
      <th>author_flair_richtext</th>
      <th>num_reports</th>
      <th>num_crossposts</th>
      <th>is_meta</th>
      <th>score</th>
      <th>upvote_ratio</th>
      <th>author_flair_css_class</th>
      <th>mod_reason_by</th>
      <th>report_reasons</th>
      <th>stickied</th>
      <th>subreddit_subscribers</th>
      <th>quarantine</th>
      <th>whitelist_status</th>
      <th>created_utc</th>
      <th>author_patreon_flair</th>
      <th>media_embed</th>
      <th>num_comments</th>
      <th>user_reports</th>
      <th>total_awards_received</th>
      <th>secure_media</th>
      <th>pinned</th>
      <th>archived</th>
      <th>treatment_tags</th>
      <th>pwls</th>
      <th>permalink</th>
      <th>ups</th>
      <th>distinguished</th>
      <th>mod_reason_title</th>
      <th>send_replies</th>
      <th>author_flair_background_color</th>
      <th>removal_reason</th>
      <th>wls</th>
      <th>domain</th>
      <th>author_flair_text</th>
      <th>removed_by_category</th>
      <th>category</th>
      <th>clicked</th>
      <th>link_flair_text</th>
      <th>approved_at_utc</th>
      <th>secure_media_embed</th>
      <th>over_18</th>
      <th>removed_by</th>
      <th>is_robot_indexable</th>
      <th>name</th>
      <th>link_flair_text_color</th>
      <th>selftext</th>
      <th>author_flair_type</th>
      <th>contest_mode</th>
      <th>link_flair_css_class</th>
      <th>allow_live_comments</th>
      <th>created</th>
      <th>media</th>
      <th>downs</th>
      <th>is_video</th>
      <th>selftext_html</th>
      <th>subreddit_name_prefixed</th>
      <th>awarders</th>
      <th>spoiler</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>text</td>
      <td>False</td>
      <td>investing</td>
      <td>t5_2qhhq</td>
      <td>False</td>
      <td>False</td>
      <td>all_ads</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>[]</td>
      <td>False</td>
      <td>jh0aqu</td>
      <td>rexmakesbeats</td>
      <td>False</td>
      <td>public</td>
      <td>NaN</td>
      <td>False</td>
      <td>t2_lj3n9</td>
      <td>Tesla Weekly Analysis - Week ending 10/24/2020</td>
      <td>True</td>
      <td>{}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://www.reddit.com/r/investing/comments/jh...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1185618</td>
      <td>False</td>
      <td>all_ads</td>
      <td>1.603503e+09</td>
      <td>False</td>
      <td>{}</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>6</td>
      <td>/r/investing/comments/jh0aqu/tesla_weekly_anal...</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>self.investing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{}</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>t3_jh0aqu</td>
      <td>dark</td>
      <td>Technical analysis on Tesla for the week.  We ...</td>
      <td>text</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.603532e+09</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>&amp;lt;!-- SC_OFF --&amp;gt;&amp;lt;div class="md"&amp;gt;&amp;lt...</td>
      <td>r/investing</td>
      <td>[]</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>text</td>
      <td>False</td>
      <td>investing</td>
      <td>t5_2qhhq</td>
      <td>False</td>
      <td>False</td>
      <td>all_ads</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>[]</td>
      <td>False</td>
      <td>jh0a9u</td>
      <td>jdybka</td>
      <td>False</td>
      <td>public</td>
      <td>NaN</td>
      <td>False</td>
      <td>t2_53g7qwfc</td>
      <td>Book review: Investing In Biotech</td>
      <td>True</td>
      <td>{}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://www.reddit.com/r/investing/comments/jh...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1185618</td>
      <td>False</td>
      <td>all_ads</td>
      <td>1.603503e+09</td>
      <td>False</td>
      <td>{}</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>6</td>
      <td>/r/investing/comments/jh0a9u/book_review_inves...</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>self.investing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{}</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>t3_jh0a9u</td>
      <td>dark</td>
      <td>Published 18 years ago, I was hesitant this bo...</td>
      <td>text</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.603532e+09</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>&amp;lt;!-- SC_OFF --&amp;gt;&amp;lt;div class="md"&amp;gt;&amp;lt...</td>
      <td>r/investing</td>
      <td>[]</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>text</td>
      <td>False</td>
      <td>investing</td>
      <td>t5_2qhhq</td>
      <td>False</td>
      <td>False</td>
      <td>all_ads</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>[]</td>
      <td>False</td>
      <td>jh08np</td>
      <td>louissanchez84</td>
      <td>False</td>
      <td>public</td>
      <td>NaN</td>
      <td>False</td>
      <td>t2_rv3pk</td>
      <td>Need some advice for Porfollios</td>
      <td>True</td>
      <td>{}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://www.reddit.com/r/investing/comments/jh...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1185618</td>
      <td>False</td>
      <td>all_ads</td>
      <td>1.603503e+09</td>
      <td>False</td>
      <td>{}</td>
      <td>1</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>6</td>
      <td>/r/investing/comments/jh08np/need_some_advice_...</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>self.investing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{}</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>t3_jh08np</td>
      <td>dark</td>
      <td>I'm willing to take some risk on all portfolio...</td>
      <td>text</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.603532e+09</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>&amp;lt;!-- SC_OFF --&amp;gt;&amp;lt;div class="md"&amp;gt;&amp;lt...</td>
      <td>r/investing</td>
      <td>[]</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>text</td>
      <td>False</td>
      <td>investing</td>
      <td>t5_2qhhq</td>
      <td>False</td>
      <td>False</td>
      <td>all_ads</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>[]</td>
      <td>False</td>
      <td>jgzoym</td>
      <td>ttyler1789</td>
      <td>False</td>
      <td>public</td>
      <td>NaN</td>
      <td>False</td>
      <td>t2_1e3atzjp</td>
      <td>New(?) investing strategy?</td>
      <td>True</td>
      <td>{}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://www.reddit.com/r/investing/comments/jg...</td>
      <td>[]</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>[]</td>
      <td>NaN</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>0.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1185618</td>
      <td>False</td>
      <td>all_ads</td>
      <td>1.603501e+09</td>
      <td>False</td>
      <td>{}</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>[]</td>
      <td>6</td>
      <td>/r/investing/comments/jgzoym/new_investing_str...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6</td>
      <td>self.investing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{}</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>t3_jgzoym</td>
      <td>dark</td>
      <td>I've posted this on r/wallstreetbets a couple ...</td>
      <td>text</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>1.603530e+09</td>
      <td>NaN</td>
      <td>0</td>
      <td>False</td>
      <td>&amp;lt;!-- SC_OFF --&amp;gt;&amp;lt;div class="md"&amp;gt;&amp;lt...</td>
      <td>r/investing</td>
      <td>[]</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Retain relevant Columns


```python
# Relevant Columns that will help in the classification process
retain_cols = ['subreddit','selftext','author_fullname','title','link_flair_css_class', 'upvote_ratio', 'ups',\
        'score','created_utc']
```


```python
df = df[retain_cols]
```


```python
df.head()
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
      <th>subreddit</th>
      <th>selftext</th>
      <th>author_fullname</th>
      <th>title</th>
      <th>link_flair_css_class</th>
      <th>upvote_ratio</th>
      <th>ups</th>
      <th>score</th>
      <th>created_utc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>investing</td>
      <td>Technical analysis on Tesla for the week.  We ...</td>
      <td>t2_lj3n9</td>
      <td>Tesla Weekly Analysis - Week ending 10/24/2020</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.603503e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>investing</td>
      <td>Published 18 years ago, I was hesitant this bo...</td>
      <td>t2_53g7qwfc</td>
      <td>Book review: Investing In Biotech</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.603503e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>investing</td>
      <td>I'm willing to take some risk on all portfolio...</td>
      <td>t2_rv3pk</td>
      <td>Need some advice for Porfollios</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.603503e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>investing</td>
      <td>I've posted this on r/wallstreetbets a couple ...</td>
      <td>t2_1e3atzjp</td>
      <td>New(?) investing strategy?</td>
      <td>NaN</td>
      <td>0.4</td>
      <td>0</td>
      <td>0</td>
      <td>1.603501e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>investing</td>
      <td>I also asked r/stocks but it's probably better...</td>
      <td>t2_4uwlv6py</td>
      <td>For those of you that invest in "big name" EV ...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>1.603499e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change created_utc into datetime data type
df['created_utc'] = df['created_utc'].map(lambda x: datetime.datetime.fromtimestamp(x,))
```


```python
df[['created_utc']].head()
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
      <th>created_utc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-24 09:35:30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-24 09:34:39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-24 09:31:43</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-24 08:54:44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-24 08:16:16</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Separate created time into month, weekday and hours
df['month'] = df['created_utc'].dt.month
df['weekday'] = df['created_utc'].dt.weekday
df['hour'] = df['created_utc'].dt.hour
```


```python
df.head()
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
      <th>subreddit</th>
      <th>selftext</th>
      <th>author_fullname</th>
      <th>title</th>
      <th>link_flair_css_class</th>
      <th>upvote_ratio</th>
      <th>ups</th>
      <th>score</th>
      <th>created_utc</th>
      <th>month</th>
      <th>weekday</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>investing</td>
      <td>Technical analysis on Tesla for the week.  We ...</td>
      <td>t2_lj3n9</td>
      <td>Tesla Weekly Analysis - Week ending 10/24/2020</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 09:35:30</td>
      <td>10</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>investing</td>
      <td>Published 18 years ago, I was hesitant this bo...</td>
      <td>t2_53g7qwfc</td>
      <td>Book review: Investing In Biotech</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 09:34:39</td>
      <td>10</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>investing</td>
      <td>I'm willing to take some risk on all portfolio...</td>
      <td>t2_rv3pk</td>
      <td>Need some advice for Porfollios</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 09:31:43</td>
      <td>10</td>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>investing</td>
      <td>I've posted this on r/wallstreetbets a couple ...</td>
      <td>t2_1e3atzjp</td>
      <td>New(?) investing strategy?</td>
      <td>NaN</td>
      <td>0.4</td>
      <td>0</td>
      <td>0</td>
      <td>2020-10-24 08:54:44</td>
      <td>10</td>
      <td>5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>investing</td>
      <td>I also asked r/stocks but it's probably better...</td>
      <td>t2_4uwlv6py</td>
      <td>For those of you that invest in "big name" EV ...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 08:16:16</td>
      <td>10</td>
      <td>5</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Text Cleaning


```python
df['story']= df['title'] + " - " + df['selftext']
```


```python
df.loc[1,'story']
```




    "Book review: Investing In Biotech - Published 18 years ago, I was hesitant this book would be outdated. It does show signs of age when David Harper gives specific examples of biotechs. In one case he mentions Nexia Biotechnologies which went bankrupt almost 10 years ago. Although there was a lot of hype around the company's product; using [genetically engineered goats](https://www.cbc.ca/news/canada/ottawa/spider-goats-display-angers-ottawa-professor-1.1137229) to manufacture BioSteel. Sounds revolutionary and high tech, which makes it a good cautionary tale. It's easy to get hyped about the products but that doesn't mean it's economically viable.\n\nThe book did have some timeless advice though. I've made a list of some points to consider when researching a biotech:\n\n1.The goals of each clinical trial:\n\nPreclinical: Animal testing\n\nPhase 1: Is it safe?\n\nPhase 2: Does it work?\n\nPhase 4: Is it safe and does it work for a lot of people?\n\n2.Be wary if the FDA is questioning the statistics of a trial. It's a possible sign the company will have to redo it. This is costly and time consuming.  \n\n3. Alliances are key. Are they partnering with large, well known pharmaceutical companies?\n\n4. Look for venture capital and IB backing. They have a team of experts so if they endorse it, it's a good sign. It's also a very positive sign if the investment bank buys shares directly in the company. Alternatively, be cautious if the biotech is raising capital alone, in which case you have to question why no one is interested. \n\n5. Look at the background of management, do they have a proven track record of regulatory approval?\n\n6. Burn rate: how fast a biotech consumes capital. Ideally, they should have 2-3 years' worth of cash on hand based on their burn rate. Companies with more cash are less likely to try to raise capital by diluting shareholders. \n\n7. Be skeptical of phase 3 trials with less than 500 participants. Ignore results when there are less than 100 participants. \n\n8. 40% rule. If bad news surfaces, you can expect the stock price to fall \\~40%."




```python
df.loc[1,'story']
```




    "Book review: Investing In Biotech - Published 18 years ago, I was hesitant this book would be outdated. It does show signs of age when David Harper gives specific examples of biotechs. In one case he mentions Nexia Biotechnologies which went bankrupt almost 10 years ago. Although there was a lot of hype around the company's product; using [genetically engineered goats](https://www.cbc.ca/news/canada/ottawa/spider-goats-display-angers-ottawa-professor-1.1137229) to manufacture BioSteel. Sounds revolutionary and high tech, which makes it a good cautionary tale. It's easy to get hyped about the products but that doesn't mean it's economically viable.\n\nThe book did have some timeless advice though. I've made a list of some points to consider when researching a biotech:\n\n1.The goals of each clinical trial:\n\nPreclinical: Animal testing\n\nPhase 1: Is it safe?\n\nPhase 2: Does it work?\n\nPhase 4: Is it safe and does it work for a lot of people?\n\n2.Be wary if the FDA is questioning the statistics of a trial. It's a possible sign the company will have to redo it. This is costly and time consuming.  \n\n3. Alliances are key. Are they partnering with large, well known pharmaceutical companies?\n\n4. Look for venture capital and IB backing. They have a team of experts so if they endorse it, it's a good sign. It's also a very positive sign if the investment bank buys shares directly in the company. Alternatively, be cautious if the biotech is raising capital alone, in which case you have to question why no one is interested. \n\n5. Look at the background of management, do they have a proven track record of regulatory approval?\n\n6. Burn rate: how fast a biotech consumes capital. Ideally, they should have 2-3 years' worth of cash on hand based on their burn rate. Companies with more cash are less likely to try to raise capital by diluting shareholders. \n\n7. Be skeptical of phase 3 trials with less than 500 participants. Ignore results when there are less than 100 participants. \n\n8. 40% rule. If bad news surfaces, you can expect the stock price to fall \\~40%."




```python
# Perform data cleaning using regular expression
df['rex_clean'] = df['story'].map(lambda x: rex(x))
```


```python
df.loc[1,'rex_clean']
```




    'book review investing in biotech published years ago i was hesitant this book would be outdated it does show signs of age when david harper gives specific examples of biotechs in one case he mentions nexia biotechnologies which went bankrupt almost years ago although there was a lot of hype around the company s product using genetically engineered goats to manufacture biosteel sounds revolutionary and high tech which makes it a good cautionary tale it s easy to get hyped about the products but that doesn t mean it s economically viable the book did have some timeless advice though i ve made a list of some points to consider when researching a biotech the goals of each clinical trial preclinical animal testing phase is it safe phase does it work phase is it safe and does it work for a lot of people be wary if the fda is questioning the statistics of a trial it s a possible sign the company will have to redo it this is costly and time consuming alliances are key are they partnering with large well known pharmaceutical companies look for venture capital and ib backing they have a team of experts so if they endorse it it s a good sign it s also a very positive sign if the investment bank buys shares directly in the company alternatively be cautious if the biotech is raising capital alone in which case you have to question why no one is interested look at the background of management do they have a proven track record of regulatory approval burn rate how fast a biotech consumes capital ideally they should have years worth of cash on hand based on their burn rate companies with more cash are less likely to try to raise capital by diluting shareholders be skeptical of phase trials with less than participants ignore results when there are less than participants rule if bad news surfaces you can expect the stock price to fall'




```python
# Lemmatizing words
df['lem_text'] = df['rex_clean'].map(lambda x: text_clean(x, mode='lem'))
```


```python
df['lem_text']
```




    0       tesla weekly analysis week ending technical an...
    1       book review investing biotech published year a...
    2       need advice porfollios willing take risk portf...
    3       new investing strategy posted r wallstreetbets...
    4       invest big name ev company like tesla nio hedg...
                                  ...                        
    2986    weekday help victory thread week august need h...
    2987    potential employer offered pay le job applicat...
    2988    coronavirus megathread update resource discuss...
    2989    coronavirus megathread update resource discuss...
    2990    coronavirus megathread update resource discuss...
    Name: lem_text, Length: 2991, dtype: object




```python
# Stemming words
df['stem_text'] = df['rex_clean'].map(lambda x: text_clean(x, mode='stem'))
```


```python
df['stem_text']
```




    0       tesla weekli analysi week end technic analysi ...
    1       book review invest biotech publish year ago he...
    2       need advic porfollio will take risk portfolio ...
    3       new invest strategi post r wallstreetbet coupl...
    4       invest big name ev compani like tesla nio hedg...
                                  ...                        
    2986    weekday help victori thread week august need h...
    2987    potenti employ offer pay less job applic appli...
    2988    coronaviru megathread updat resourc discuss qu...
    2989    coronaviru megathread updat resourc discuss qu...
    2990    coronaviru megathread updat resourc discuss qu...
    Name: stem_text, Length: 2991, dtype: object



### Tagging of subreddit


```python
df['reddit_tag'] = df['subreddit'].map({'investing': 1, 'personalfinance':0})
```


```python
df['reddit_tag'].value_counts(normalize=True)
```




    0    0.635573
    1    0.364427
    Name: reddit_tag, dtype: float64




```python
df['story_lem_len'] = df['lem_text'].str.len()
```


```python
df.shape
```




    (2991, 18)




```python
# Look for story that have at least 255 characters.
df = df[df['story_lem_len']>255]
```

<span style= "color:magenta">Remarks:</span>  
The tagging is to create a column which can be use as the Y-variable for training the model.  
The next step is to filter posts that have less than 255 characters. The 255 character limit is an arbitrary numbers which I determined based on a quick look at most of the posts. Posts that are below this limits tend to contains little to no information or hyperlinks that redirects users to other sites.


```python
df.shape
```




    (2264, 18)




```python
df.head(2)
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
      <th>subreddit</th>
      <th>selftext</th>
      <th>author_fullname</th>
      <th>title</th>
      <th>link_flair_css_class</th>
      <th>upvote_ratio</th>
      <th>ups</th>
      <th>score</th>
      <th>created_utc</th>
      <th>month</th>
      <th>weekday</th>
      <th>hour</th>
      <th>story</th>
      <th>rex_clean</th>
      <th>lem_text</th>
      <th>stem_text</th>
      <th>reddit_tag</th>
      <th>story_lem_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>investing</td>
      <td>Technical analysis on Tesla for the week.  We ...</td>
      <td>t2_lj3n9</td>
      <td>Tesla Weekly Analysis - Week ending 10/24/2020</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 09:35:30</td>
      <td>10</td>
      <td>5</td>
      <td>9</td>
      <td>Tesla Weekly Analysis - Week ending 10/24/2020...</td>
      <td>tesla weekly analysis week ending technical an...</td>
      <td>tesla weekly analysis week ending technical an...</td>
      <td>tesla weekli analysi week end technic analysi ...</td>
      <td>1</td>
      <td>519</td>
    </tr>
    <tr>
      <th>1</th>
      <td>investing</td>
      <td>Published 18 years ago, I was hesitant this bo...</td>
      <td>t2_53g7qwfc</td>
      <td>Book review: Investing In Biotech</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>2020-10-24 09:34:39</td>
      <td>10</td>
      <td>5</td>
      <td>9</td>
      <td>Book review: Investing In Biotech - Published ...</td>
      <td>book review investing in biotech published yea...</td>
      <td>book review investing biotech published year a...</td>
      <td>book review invest biotech publish year ago he...</td>
      <td>1</td>
      <td>1252</td>
    </tr>
  </tbody>
</table>
</div>


### Word Cloud
**Word Cloud of Investing Corpus**
![Investing Word Cloud](/images/portfolio/p003_project_evian/InvestingWordCloud.png)

<span style= "color:magenta">Remarks:</span> A quick look at the word cloud show that most of the posts that are classified as investing have high frequency words relating to investing eg. stock, company, market, share. This give an indication that majority of posts has been posted correctly in the respective subreddits.

**Word Cloud of Personal Finance**
![Investing Word Cloud](/images/portfolio/p003_project_evian/PersonalFinanceWordCloud.png)

<span style= "color:magenta">Remarks:</span>  A quick look at the word cloud show that most of the posts that are classified as investing have high frequency words relating to investing eg. account, credit, loan, income, mortgage. However, there are some words such as year & month that are high in term of frequncy among all the posts. As these words may be relating to terms ~ payment term , they will need to be removed since they will likely appear in both personal finance & investing subreddits.

## Modeling of Lem text

<span style= "color:magenta">Remarks:</span> Both Stemming and Lemmatization both generate the root form of the inflected words. The difference is that stem might not be an actual word whereas, lemma is an actual language word.Stemming follows an algorithm with steps to perform on the words which makes it faster. Whereas, in lemmatization, you used WordNet corpus and a corpus for stop words as well to produce lemma which makes it slower than stemming. Based on this, Lemmatized Text is used for modeling as it retain the characteristic of the posts better than stemming.

### Baseline Model
```python
y_lem.value_counts(normalize=True)
```




    0    0.640901
    1    0.359099
    Name: reddit_tag, dtype: float64


## Modelling using Count Vectorizer | TF-IDF with Log Regression | Naive Bayes

<span style= "color:magenta">Remarks:</span> Looking at the users' posts that I have collected, the count of the posts will be a baseline in which I can evluate against my classification models that I will be created. Looking at the ratio of the based like, it is roughly 65% - 35%, the proportion is not too skewed, best case is 50:50, but  I will be stratifying my train-test-split to ensure that the models are trained properly.


```python
# Adding pre-identifed stopwords to CountVector STOPWORDS.
stop_words = text.ENGLISH_STOP_WORDS.union(['year', 'month'])
```

<span style= "color:magenta">Remarks:</span> From the wordcloud we did identify 'year' & 'month' are words that do not help in the models, these will be added to CountVectorizer's stopwords.  
A new list is created to hold the union of the StopWords. 

*Stopwords are list of words which we explicitly tell the algorithms not to include in the modelling. These words are usually specific to the context of the dataset or problems we are trying to solve*

**Coefficient of Lem Corpus using Count-Vectorizer with Logistic Regression Modelling**
![CoefficientOfWords](/images/portfolio/p003_project_evian/CoefficientOfWords_LogReg_cv.png)

<span style= "color:magenta">Remarks:</span> Looking at the top 60 words with the highest frequency, there are alot of words which cannot assist to classify the subreddit. As such, these words will be removed from the model. An arbitrage cutoff of 1 ~ words that are of no predictive value with coefficient value of >1 will be removed as they will have bigger impact on the log regression which we want to minimize.

### Evaluating Log Regerssion with Count Vectorizer

```python
# Train Model accuracy
pipe_log_2.score(X_lem_train, y_lem_train)
```
    0.9089709762532981

```python
# Test accuracy
pipe_log_2.score(X_lem_val, y_lem_val)
```
    0.8770053475935828


<span style= "color:magenta">Remarks:</span> Our model's accuracy for the evaluation dataset has neither improve or deteriorate greatly against the training dataset, suggesting that removing custom stop words is not a necessary process for this dataset.


![LogReg_cv_ROC](/images/portfolio/p003_project_evian/LogReg_cv_ROC.png)

<span style= "color:magenta">Remarks:</span> Looking at the ROC curve, with a AUC socre of 0.94 which is very close to 1. It give me confident that the model is working well against the evaluation data set. The max score of AUC scorce is 1.

![LogReg_cv_CMatrix](/images/portfolio/p003_project_evian/LogReg_cv_CMatrix.png)

<span style= "color:magenta">Remarks:</span> Looking at the confusion matrix, our model is able to class user's comment quite accuracy @ 88%, this is above our agreed thresoldof 80%.  
We do notice there there is about 12% errors in our model (Type I & Type II). Upon Investigation, I realized that most of these errors are resulted from users themselves not able to clearly defined their posts as being personal finance or investing.  
As mentioned in the introduction, this is an expected concern as investing are very closely related to personal finance and users might not be to "class" their posts accordingly. Besides, the errors, especially type II, are not an issues to our client, Seeking Alpha, as it may means unexpected customers.

### Evaluating Log Regerssion with TF-IDF
```python
# Train Model accuracy
pipe_log_reg_tfidf_2.score(X_lem_train, y_lem_train)
```
    0.908311345646438

```python
# Test accuracy
pipe_log_reg_tfidf_2.score(X_lem_val, y_lem_val)
```
    0.8850267379679144


<span style= "color:magenta">Remarks:</span> By removing words using custom stop words, our model accuracy have dropped by 0.2%. Again, there is not much improvement or deteoriation in term of accuracy and may suggest that removing custom stop words is not necessary for this dataset.

![LogReg_tfidf_ROC](/images/portfolio/p003_project_evian/LogReg_tfidf_ROC.png)

<span style= "color:magenta">Remarks:</span> Looking at the ROC curve, with a AUC socre of 0.95 which is very close to 1. It give me confident that the model is working well against the evaluation data set. The max score of AUC scorce is 1.

![LogReg_tfidf_CMatrix](/images/portfolio/p003_project_evian/LogReg_tfidf_CMatrix.png)

<span style= "color:magenta">Remarks:</span> Looking at the confusion matrix, our model is able to class user's comment quite accuracy @ 89%, this is above our agreed threshold of 80%.  
We do notice there there is about 11% errors in our model (Type I & Type II). Upon Investigation, I realized that most of this errors are resulted from users themselves not able to clearly defined their posts as being personal finance or investing.  
Again, the errors, especially type II, are not an issues to our client, Seeking Alpha, as it may means unexpected customers.

The TF-IDF Vectorizer works better compared to the CountVectorizer in this context. Across all metrics used,both the confusion matri and ROC score, TF-IDF Model has scored higher than Count Vec Model. 

### Evaluating Naive Bayes with Count Vectorizer

```python
# Score our model on the training set.
pipe_nb_cv_2.score(X_lem_train, y_lem_train)
```
    0.8852242744063324


```python
# Score our model on the testing set.
pipe_nb_cv_2.score(X_lem_val, y_lem_val)
```
    0.8676470588235294

<span style= "color:magenta">Remarks:</span> Our model accuracy differs only about 0.02% between training and evaluation dataset.

![Nb_cv_ROC](/images/portfolio/p003_project_evian/Nb_cv_ROC.png)

<span style= "color:magenta">Remarks:</span> Looking at the ROC curve, with a AUC socre of 0.93 which is very close to 1. It give me confident that the model is working well against the evaluation data set. The max score of AUC scorce is 1.

![Nb_cv_CMatrix](/images/portfolio/p003_project_evian/Nb_cv_CMatrix.png)

<span style= "color:magenta">Remarks:</span> Looking at the confusion matrix, our model is able to class user's comment quite accuracy @ 87%, this is above our agreed threshold of 80%. We do notice there there is about 13% errors in our model (Type I & Type II). As mentioned, the errors, especially type II, are not an issues to our client, Seeking Alpha, as it may means unexpected customers.

### Evaluating Naive Bayes with TF-IDF
```python
# Train Model accuracy
pipe_nb_tfidf_2.score(X_lem_train, y_lem_train)
```
    0.883245382585752

```python
# Test accuracy
pipe_nb_tfidf_2.score(X_lem_val, y_lem_val)
```
    0.8729946524064172

<span style= "color:magenta">Remarks:</span> Our model accuracy differ by about 0.1% between training and evaluation dataset.

![Nb_cv_ROC](/images/portfolio/p003_project_evian/Nb_tfidf_ROC.png)

<span style= "color:magenta">Remarks:</span> Looking at the ROC curve, with a AUC socre of 0.94 which is very close to 1. It give me confident that the model is working well against the evaluation data set. The max score of AUC scorce is 1.

![Nb_cv_ROC](/images/portfolio/p003_project_evian/Nb_tfidf_CMatrix.png)

<span style= "color:magenta">Remarks:</span> Looking at the confusion matrix, our model is able to class user's comment quite accuracy @ 87%, this is above our agreed threshold of 80%. We do notice there there is about 13% errors in our model (Type I & Type II). Per above, the errors, especially type II, are not an issues to our client, Seeking Alpha, as it may means unexpected customers.

In the case of Naive Bayes, The the 2 models performed almost on par with each other.

## Conclusion & Recommendation 

**Metric Evaluation**  

At a high-level evaluation, the 4 models are predicting correctly at least 87-89% of the time. This has exceeded the minimum requirement of the model to be at least 80%. Also, the precision, among what is predicted true, how many are true, are at least 81%; suggesting that all three models work well for the clients. At a deeper level. the three models' sensitivity and specificity are also high on top of the fact that the prevalence rate is low ~ 34. Hence we can confidently say that our models are robust in classifying unseen text (new data) correctly.

**ROC Evaluation**  

The ROC for both models is at least 0.93 out of a max of 1. This also gives a perspective that the model works well in classifying the posts correctly.

**Recommendation**  

Given the business context of the entire project, I would suggest client use the TF-IDF as the text-processing method and Naive Bayes model even though based on the metrics (Confusion Matrix and ROC) Log Regression is better than Naive Bayes. The reasons, although not exhaustive, are as follows:  

Naive Bayes works well even when there are small training data set compared to logistic regression, this can be the case in the financial market where new trading vehicles are created all the time. This will allow clients to still relative accurately identify target users who may be posting about emerging trade ideas (very few posts)    

Naive Bayes is highly scalable and also work well as the number of predictors increase. This also a likely case as the business evolves, other forms of data are available and Naive Bayes can handle such increments well as compared to Logistic Regression.  

TF-IDF is a score that tells us which words are important to one document, relative to all other documents. Words that occur often in one document but don't occur in many documents contain more predictive power. This has an edge over Count Vectorizer which only rank the importance of a word based on its frequency.  

TF-IDF also works better in the context of financial markets. There are too many products available hence the frequency of a keyword occurring in every post is minimum. As TF-IDF will give more weight to less occurring words, this will allow our client, Seeking Alpha, to pick up upcoming trend (eg. new bitcoins, product) in advance and position their marketing accordingly.

**Consideration**  

Examining the Type I & Type II errors where the models classified wrongly, the errors are generally due to users not able to articulate or word their posts correctly. This is understandable as not all users are a finance professional and due to the similarity of the topic where at times, or increasingly investing are considered a component of personal finance; Further 'cleaning' of model vocabulary might not justify the resource allocated to it given the low number of errors that occurs. One business solutions are to consider users who are classified wrongly as potential customers to market to assuming the chance of converting them into potential clients outweighs their complaints of receiving spams marketing.  

As a value add - Client can consider their marketing efforts on Monday - Thursday from 9 pm to 12 midnight as based on the post frequency; these days and times have the highest.

### Supplementary Charts
![Supplementary Charts](/images/portfolio/p003_project_evian/SupplementaryCharts.png)



