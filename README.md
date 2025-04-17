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
### HMM Backbone
  #### Part 1 - Obtaining Datas and Identify Basic Relationships
  - Features input for visualisation (from distinct endpoints, merged)
    <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; background-color: rgba(129,212,226,255); font-family: monospace;">
    Type something here...
    </div>
    > something
  - Correlation tables between features
  - Frequency plots against features
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

  ## Next
  - Parameter tuning for optimal number of regimes
3. NLP Support
4. LSTM_CNN OHLCV
5. Data Manipulation
6. HMM NLP Signals Integration
7. Features Engineering
8. Weightage Control and Application
9. Backtesting and ForwardTesting


  
