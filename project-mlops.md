# Business Challenge: Predicting BTC/USD Prices Using MLOps on AWS

## Overview

### History and Context

Cryptocurrencies have revolutionized the financial markets since the advent of Bitcoin in 2009. Bitcoin, the first decentralized digital currency, has seen explosive growth in both value and popularity. As cryptocurrencies have become more mainstream, the ability to predict their price movements has become increasingly valuable. Financial institutions, traders, and investors all seek to leverage predictive models to gain a competitive edge in the market.

### Business Need

In the highly volatile cryptocurrency market, accurate price prediction can lead to significant financial gains and risk management improvements. A model that can predict Bitcoin to USD (BTC/USD) prices on an hourly basis can be utilized by traders to make informed decisions, by financial analysts to understand market trends, and by automated trading systems to execute trades efficiently.

## Challenge Outline

### Step 1: Connect to an API to Retrieve Cryptocurrency Information

#### Objective
Connect to an API to fetch historical BTC/USD price data on an hourly basis.

### Step 2: Build a Model to Predict BTC/USD on an Hourly Basis

#### Objective
Build a machine learning model to predict BTC/USD prices using historical data. The model should be able to handle the high volatility and noise typical of cryptocurrency markets.

### Step 3: Deploy the Model Using MLOps Concepts on AWS

#### Objective
Deploy the trained machine learning model on AWS using DevOps and MLOps principles. Below you have a list of requirements/questions you should try to answer. Even if you cant answer them all, push as far as you can.

#### Goals:
 - **TIER 1**
    - Create a repository for this project
    - Create a Virtual Environment for this project
    - Deploy the code using GIT
 
 - **TIER 2**
     - Track your experiments using MLFlow
     - Run the model on the cloud (SageMaker / EC2 for example). A couple of alternatives:
        - Run it in a SageMaker
        - Use the inference endpoints of SageMaker
        - Run it in a EC2 Linux Ubuntu Server
    - Create a REST API using Flask or FastAPI to serve model predictions.
    - Set up a pipeline to automate the collection and preprocessing of currency or Bitcoin price data.
    - Retrain your model, as performance drops from a certain threshold
    
 - **TIER 3**
    - Use Docker to containerize the machine learning model and its dependencies for consistent deployment

 - **TIER4**
    - Automate the retraining of your model, as performance drops from a certain threshold

 - **BONUS**
    - Create an alerting system (email, SMS, Telegram messages) to alert you about prices going up or down
    - Implement monitoring and logging for the deployed model using AWS CloudWatch, Prometheus, or Grafana to track performance and usage.
 
 - Explore and include in future work:
    - How to implement CI/CD pipelines using tools like GitHub Actions, Jenkins, or CircleCI to automate testing and deployment processes?
    - How to monitor and manage cloud resource usage to optimize costs using AWS Cost Explorer or similar tools?
    - How to implement tools like SHAP or LIME to provide model explainability and insights into predictions?
    - What are Unit and Integration tests?
    - Explore ways to design the deployment to be scalable, using AWS services like Auto Scaling Groups or Kubernetes (EKS) for handling increased load.


### Summary

This challenge guides you through the process of fetching cryptocurrency data, building a predictive model, and deploying it using MLOps principles on AWS. By completing this challenge, you will gain practical experience in data retrieval, machine learning, and model deployment in a cloud environment.

### Materials

1. **API providers**:
    - https://min-api.cryptocompare.com/
    - https://www.binance.com/pt - only if you already have an account (it might take a few days to validate your identity)
    - https://docs.coincap.io/
    - https://coinpaprika.com/api/pricing/

**NOTE** If you struggle with obtaining the data, please move to the same exercise using any form of currency with the Yahoo Finance API [Source 1](https://github.com/ranaroussi/yfinance) [Source 2](https://pypi.org/project/yfinance/)

