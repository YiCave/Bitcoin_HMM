# Project Title: ON-CHAIN DATA AUTOMATED TRADING MACHINE
## Overview
This project focuses on applying statistical-based Hidden Markov Models(HMM), Natural Language Processing(NLP) that act as an indicators/ filters that enhanced the entire backtest process, in order to boost profitability by hitting higher sharpe ratio while achieving the criteria for maximum drawdown and trade frequency.

## Tools Used
- Python
- Jupyter Notebook
- Multiple machine learning Libraries
- Graphing Libraries
- Statistical tools
  
## Architecture Workflow

### HMM Backbone
  #### Part 1 - Obtaining Data and Identifying Basic Relationships
  - Features input for visualisation (from distinct endpoints, merged)
    
    ```
     📌 Data pre-stored in CSV to reduce redundant read operations
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
     📌 Analyse the relationship between features
     📌 Visualised using heatmaps for feature prioritisation
    ```
    
  - Frequency plots against features
    
     ```
     📌 Identifying the norm of the crypto, a significant signal for upcoming actions
       - Verify whale dominations 
       - Verify whale density
       - Verify the presence of speculative bubbles
     📌 Enable data-driven decision-making through graph shapes
       - Ensure the active trading environment
    ```
  
  #### Part 2 - Optimising Model Selection
  - Model selection using Bayesian Information Criterion, Akaike Information Criterion 
     ```
        Statistical approach:
        n_features = X_scaled.shape[1]
        n_params = n_states * n_states + n_states * n_features + n_states * n_features * (n_features + 1) // 2
        log_likelihood = model.score(X_scaled)
        bic = -2 * log_likelihood + n_params * np.log(X_scaled.shape[0])
        aic = -2 * log_likelihood + 2 * n_params
    ```
  - Silhouette score to evaluate the quality of clustering
    ```
       silhouette = silhouette_score(X_scaled, hidden_states)
       silhouette_scores.append(silhouette)
    ```
  - Elbow Plots which visualize the minimization process, state prediction
    ```
      📌 Append BIC and AIC score to an array
      📌 Identify the model(distinct number of states)that have lowest relative score
      📌 Visualise Silhouette score to identify how well data fits their cluster
    ```
  - Regime Classification and Distribution Plots
    ```
      📌 Statistically backed regime classification for data
      📌 5-year time series datapoint Visualisation
         - Enable us to easily identify data with extreme flow means
         - Better understanding of regime characteristics
      📌 Visualise Silhouette score to identify how well data fits their cluster
    ```
  - Summary Metrics for Further Regime Characteristic Identifications
    ```
    Example(using flow mean metric):    
    regime         mean         std        min          max                                                            
      0         7.288524    3.084889   0.006716    15.920316   
      1        20.494636    6.007822   0.089733    38.411333      
      2       216.062946  352.265007   0.023389  4447.247543   
      3         2.868473    1.706804   0.000000     7.368699      
      4        53.256369   20.854317  24.866651   144.325671     
      5        13.094486    8.696826   0.064580    95.027511

    📌 Did for all metrics as above
    📌 Bar Chart Graphs Visualisation to identify outlying regimes for each metric
    📌 Obtain Insights for future in-depth regime analysis
    ```
  #### Part 3 - Regime Transition Handling
  - Keeptrack regime transition by count
     ```
      Regime transition counts:
          To   0     1    2     3     4     5
      From                                   
      0       7312  2710  349  4235  1884  2012
      1       3003  2206  137  1609  1153  1130
      2        276   174  153   152   242   177
      3       3813  1481  214  2980  1245  1075
      4       1910  1291  183  1174  1655   767
      5       2188  1376  138   659   801  1320
        
      📌 Verify how likely systems moves from one regime to another
      📌 Powerful tool during the backtest period
      📌 Empowers prediction by forecasting regimes, identifying dominant transition paths
      ```
  - Convert to regime transition probability
     ```
      📌 Similar to counts, easier to use for probabilistic modeling or as transition matrices in HMMs
     ```
  - Regime Interpretation based on metrics
    ```
      📌 Each regime is characterized by statistical properties: Stability, Duration, Frequency
      📌 Enables semantic understanding, the representation of the regime's plain text meaning
      📌 Essential for drawing meaningful insights from clusters
    ```
  #### Part 4 - Transition Precision Simulation
   - Train and Test Set Splitting from Original Datasets （supervised learning)
     ```
      📌 The original dataset splitted into two, train set and test set
      📌 Learn how well a model can identify the current regime 
     ```
   - Visualisations to indicate precision
       ```
      📌 Time series plots overlaid with predicted vs. true regimes
      📌 Learn how well a model can identify the current regime 
     ```
   - Regime Prediction Accuracy
       ```
      📌 Match predicted regime sequences with actual using mapping logic
      📌 Compute aligned accuracy score to evaluate label consistency
      📌 Helps quantify model performance across all regimes (not just majority class)
     ```
   - Regime Transition Detection Accuracy
       ```
      📌 Focus on identifying regime *switch points* (transitions), not just static regimes
      📌 Evaluate using precision, recall, and F1 for transition detection 
     ```

 
### NLP Support
Matt do here

4. LSTM_CNN OHLCV
5. Data Manipulation
6. HMM NLP Signals Integration
7. Features Engineering
8. Weightage Control and Application
9. Backtesting and ForwardTesting
 - Parameter tuning for optimal number of regimes

  
