# Project Title: ON-CHAIN DATA AUTOMATED TRADING MACHINE
## Overview
This project focuses on applying statistical-based Hidden Markov Model(HMM), Natural Language Processing(NLP) that act as an indicators/ filters that enhanced the entire backtest process, in order to boost profitability by hitting higher sharpe ratio while achieving the criteria for maximum drawdown and trade frequency.

## Tools Used
- Python
- Jupyter Notebook
- Multiple machine learning Libraries
- Graphing Libraries
- Statistical tools
  
## Architechture Workflow
 <details>
      <summary>Click to expand</summary>

      Here's some hidden content!

      - Point A
      - Point B

    </details>
### HMM Backbone
  #### Part 1 - Obtaining Datas and Identify Basic Relationships
  📌 Features input for visualisation (from distinct endpoints, merged)
    ```
    📌 Prioritising exchange flow endpoints like flow_mean, flow_total and transatction_count,
        also includes inflow, outflow and netflow endpoints by utilising formulas. Example:
        👉 f_ttl = concat(r1,r2) +  exponential noise
        👉 f_mean= f_ttl / uniform distributor(10-30)
        👉 t_cnt= f_ttl * rand(0.5 - 2) + base offset
    📌 Merging into a CSV file through initial data preprocessing (splitted hourly, 5 years' data)
    📌 Prevent redundant request to read data each time
    📌 Enhancing
    ```
  📌 Correlation tables between features
   
  📌 Frequency plots against features
  #### Part 2 - Optimising Model Selection
  - Model selection using BIC (Bayesian Information Criterion), AIC (Akaike Information Criterion) and Silhouette score
  - Statistical approach on choosing optimised model
  - Elbow Plots which visualize the minimisation process, States prediction
  - Regime Classification and Distribution Plots
  - Statistial-based Regime Analysis
  #### Part 3 - Regime Transition Handling
  - Keeptrack regime transition by count
  - Probabilities Conversion
  - Correlation tables between regimes (from and to)
  - Regime Interpretation based on features
  #### Part 4 - Transition Precision Simulation
   - Train and Test Set Splitting from Original Datasets （supervised learning)
   - Visualisations to indicate precision
   - Regime Prediction Accuracy
   - Regime Transition Detection Accuracy

 
### NLP Support

4. LSTM_CNN OHLCV
5. Data Manipulation
6. HMM NLP Signals Integration
7. Features Engineering
8. Weightage Control and Application
9. Backtesting and ForwardTesting
 - Parameter tuning for optimal number of regimes

  
