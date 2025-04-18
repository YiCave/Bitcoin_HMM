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
    
    ```
     📌 Data pre-stored in CSV to reduce redundant read operation
     📌 Exchange flow endpoints prioritized: flow_mean, flow_total, transaction_count
     📌 Additional metrics: inflow, outflow, netflow 
     📌 Handling missing values and synthetic data generation:
         👉 f_ttl = concat(r1, r2) + exponential noise
         👉 f_mean = f_ttl / uniform(10–30)
         👉 t_cnt = f_ttl * rand(0.5–2) + base offset
     📌 Aggregated into hourly intervals, covering 5 years
     📌 5-year visualisation using timeseries plots
    ```
    
  - Correlation tables between features
    
    ```
     📌 Generate correlation matrix (0<=x<=1) among features
     📌 Analyse relationship between features
     📌 Visualised using heatmaps for feature prioritisation
    ```
    
  - Frequency plots against features
    
     ```
     📌 Identifying norm of the crypto, significant signal for upcoming actions
       - Verify whale dominations 
       - Verify whale density
       - Verify speculative bubbles' presence
     📌 Enable data-driven decision-making through graph shapes
       - Ensure active trading environment
    ```
  
  #### Part 2 - Optimising Model Selection
  - Model selection using Bayesian Information Criterion, Akaike Information Criterion and Silhouette score
     ```
        n_features = X_scaled.shape[1]
        n_params = n_states * n_states + n_states * n_features + n_states * n_features * (n_features + 1) // 2
        log_likelihood = model.score(X_scaled)
        bic = -2 * log_likelihood + n_params * np.log(X_scaled.shape[0])
        aic = -2 * log_likelihood + 2 * n_params
    ```
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

  
