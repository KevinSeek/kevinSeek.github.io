---
layout: post
title:  "Recommender System for Almost Anything "
date:   2020-11-02 10:00:00 +0800
categories: ['Classification Techniques', Ensemble Modelling, Web-Scraping, Data-Cleaning, Word Cloud, Neural Network]
excerpt_separator: <!--more-->
---


## Introduction
This is a graduation project done as part of Data Science Immersive Course conducted by General Assembly. This project serves as a stepping stone to explore deeper into the techniques used in recommender system and to showcase the following skillsets:
- Conceptualization thinking
- Web Scraping
- Neural Network Modelling
- Ensemble Modelling

## CodeBook
The full codebook can be found [here](https://github.com/KevinSeek/Kevin_Project_Portfolio/tree/master/P005_recommending_almost_anything)
<!--more-->

## Scenario

As techonology progress and more of our lifestyle revolving around tech gadgets, many companies have accumulated a large amount of data pertaining to their customers. The key question has and always been - "How are we going to tap on this immerse wealth of data not just to gain insight into both our customers and our own strengths but also to monetize it?"  

The simplest of solution or the most practical will be to hire a team of data scientists, data dump and help me solve my problem. Yet, very often, whether intentionally or unintentionally, stakeholder left out the most crucial piece of information - full access of domain knowledge. Due to this missing chunk, data scientists often and usually create analysis or products that are sub-par to the expectation of the stakeholder. More often than more, data scientist spend much time gathering these domain knowledge from various stakeholders that little time is left to produce quality and succinct analysis and product.

As a junior data scientist, I am hired by BeyondHuman Research Center to perform an analysis on recommender system and whether it is possible to create a recommender system which can assist users to navigate through the enormous data stockpile monetize it. I am provided access to limited but substantial amount of data to quickly (~approx 2 weeks) generate ways to help business stakeholders make better sale decisions yet with little or no interactions with any of the stakeholders. Beyond this scope and restrictions, I am given limited time to tap on the knowledge of other tech teams in the course of this project.

The simulated datasets used is Kaggle Netflix Prize Dataset. The data is sufficient large (~4GB) but limited in information - no user's profile except their ratings, title details are missing except for the release year.

## Problem Statement

Using the Netflix Dataset, develop recommendation system where new customers are able to find titles pertaining to their taste out of all the titles available in the repository. The recommendation system need to fulfill the following criteria:

- To design and create a robust recommendation systems where titles are pushed to users by knowing their preferences/tastes in doing so eliminate cold-start problems that have plague most early recommendation system.
- Recommendation system needs to highlight and persuade customers why the selections are recommended to them.

## Executive Summary

Using Netflix as an example to simulate situation where company or department are holding to immersive 'data wealth' but do not have the ability to monetize it (depending on context, either in producing profit or gain further insights/efficiencies).

Often insights or in this case recommendation are pushed to customers based on prior knowledge or widely accepted facts - eg. titles that are popular or knowing the customer personally. This create a knowledge silo or key-person risks where a single person or system may hold the company hostage simplify because of it being indispensable. In addition, this may that recommendations are 'whole-sale' and changing trends from company-push to on-demand model; customers may be left unsatisfied if a company do not provide a personalized touch to their experience.

Using multiple recommendation systems which targets user's rating similarity; title-similarity and looking for relationship between titles based on their meta-data seems like a good way to start to personalized products to customers.

Based on the model created, we can see that all recommendations systems provide different titles based on different aspect of the understanding the user. An ensemble model was created to find the best title recommendation from each of the sub-recommendation system. This will give a more generalized titles recommendations to user.

Comparing to the base recommendations - popularity titles, we see that our recommendations systems pushes titles that are more relating to the user's preference. In addition to this, the systems in its entirety has achieved to eliminate cold start problem where new users may be recommended something that are not to their taste.

### Key Observations:

Limitation of data available. Initially API-calls to MovieDB was supposed to be the panacea to the limited information available. However, MoiveDB do not have an exact match function but give the best movie hence I have to discards some of my data. In total, I have to reduce close to about 16% of my data.

The traditional machine learning models based recommendation systems - Collaborative Filtering and Title-Based are computationally expensive. This limits me to use a subset in order to generate the user matrix. Hence the recommendations may not be as comprehensive in terms of the titles to push to users. In additional, for every user, it need to re-calculate the user matrix again hence reducing the user experience when they need to wait longer time to find titles that are suitable to their taste.

*Neural Network (Meta-data rec system) although eliminate the problems experienced by traditional ML but it suffers from longer training time when we runs more epochs and more embedded layers. However, it do not suffers the long calculation time required to generate recommendations to users.

Each systems have their pros and cons and the bottlenecks occurs at different sections. However, storing preference is for Neutral Networks because:
- Heavy lifting is during the model training phase hence the on the user-front, there is close to little lag in pushing titles to users.
- Additional of inputs/variables are an additional layers into the Neural Network. It is relative easier to perform features "top-up" compared to traditional MLs.

### Recommendations & Further Research
Given the short turn-around time, evaluation of the recommendation system is limited to referencing to the popular titles in the database. The next steps is to explore deeper into evaluation metrics using confusion matrix metric to find how each system are recommending titles correctly using created training datasets (created as part of training for neural network)

Another direction is to look into TDIDF of meta-data. This will give a aggregated weight matrix of meta-data across all the titles. Using this matrix in the Neural Network might give better recommendation as it takes into account of weight of each keyword the title contains. Currently, each keywords is assumed to have the same weight.


* Neural Network Concept & use are heavily inspired by the following article:
<a href="https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9">Building a Recommendation System Using Neural Network Embeddings</a>