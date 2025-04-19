# Project Title: ON-CHAIN DATA AUTOMATED TRADING MACHINE
## Overview
This project focuses on applying statistical-based Hidden Markov Models(HMM), Natural Language Processing(NLP) that act as indicators/ filters that enhance the entire backtest process, in order to boost profitability by hitting a higher sharpe ratio while achieving the criteria for maximum drawdown and trade frequency.

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
      📌 Visualise Silhouette score to identify how well the data fits the cluster
    ```
  - Regime Classification and Distribution Plots
    ```
      📌 Statistically backed regime classification for data
      📌 5-year time series datapoint Visualisation
         - Enable us to easily identify data with extreme flow means
         - Better understanding of regime characteristics
      📌 Visualise Silhouette score to identify how well the data fits their cluster
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
        
      📌 Verify how likely systems move from one regime to another
      📌 Powerful tool during the backtest period
      📌 Empowers prediction by forecasting regimes, identifying dominant transition paths
      ```
  - Convert to regime transition probability
     ```
      📌 Similar to counts, easier to use for probabilistic modelling or as transition matrices in HMMs
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
      📌 The original dataset was split into two: the train set and the test set
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
      📌 Helps quantify model performance across all regimes (not just the majority class)
     ```
   - Regime Transition Detection Accuracy
       ```
      📌 Focus on identifying regime *switch points* (transitions), not just static regimes
      📌 Evaluate using precision, recall, and F1 for transition detection 
     ```

 
### NLP Support
  #### Part 1 - Preprocessing and Feature Engineering
   - Input preprocessing and representation
     ```
     📌 Data loaded from structured CSV or DataFrame  
     📌 Pre-labelled with sentiments: negative, neutral, positive  
     📌 Normalization includes lowercasing, punctuation removal  
     📌 Token and word-level handling for varied sources (tweets, news, posts)
     ```
   - TF-IDF Feature Construction
     ```
     📌 Use TF-IDF Vectorizer with unigram and bigram support  
     📌 Converts text to sparse numerical vectors  
     📌 Ideal for small-to-medium datasets  
     📌 Sample weighting is used to address class imbalance  
     ```
   - Label Encoding
     ```
     📌 Sentiment class labels mapped for modelling:
        👉 ‘negative’ → 0
        👉 ‘neutral’  → 1
        👉 ‘positive’ → 2
     📌 Consistent across both traditional ML and BERT pipelines  
     ```
  ### Part 2 - Model Building and Optimisation
   - Logistic Regression with TF-IDF
     ```
     📌 Classic ML pipeline: TF-IDF + Logistic Regression  
     📌 Class weights handled to reduce bias from imbalanced data
     📌 Fast training, interpretable coefficients  
     📌 Good baseline for resource-constrained environments  
     ```
   - BERT (DISTILBERT) Fine-Tuning
     ```
     📌 Tokenization via HuggingFace `DistilBertTokenizer`  
        👉 Pad and truncate to 128 tokens  
        👉 Attention masks generated  
     📌 Sentiment classification head with 3 output neurons  
     📌 Fine-tuning parameters:  
        👉 Mixed precision (FP16)  
        👉 Gradient accumulation  
        👉 Epoch-based model checkpointing  
     ```
   - Metrics for Evaluation
     ```
     📌 Evaluation using:
        👉 Accuracy Score  
        👉 Weighted F1 Score  
     📌 Track performance on the test set per epoch  
     📌 Highlighted per model: traditional vs. transformer-based  
     ```
  ### Part 3 - Comparative Visualisation and Analysis
   - Score Tracking and Comparison
     ```
     📌 Evaluation matrix for both models:
         - Accuracy comparison (TF-IDF vs. BERT)  
         - F1 score: highlighting class prediction quality 
     ```
  ### Part 4 - Future Directions and Production-Level Strategy
   - Domain-Specific Model Integration
     ```
     📌 Integration of `ProsusAI/finbert` for financial language  
     📌 Domain-tuned BERT boosts performance in crypto finance context  
     📌 Transfer learning from FinBERT to enhance downstream tasks  
     ```
   - Hybrid Ensemble Model (Future Work)
     ```
     📌 Combine predictions from:
        - TF-IDF model (fast, stable)
        - BERT model (deep, contextual)
     📌 Use ensemble logic or meta-classifier for final decision     
     ```
   - Deployment Considerations
     ```
     📌 TF-IDF pipeline suitable for lightweight deployment  
     📌 BERT pipeline suited for heavyweight deployment (the better model)
     📌 REST API with Flask/FastAPI backend for real-time sentiment scoring  
     ```
     
### 4. LSTM_CNN OHLCV
### 5. Data Manipulation
#### Part 1 - Initial Data Standardization and Cleaning
    - Loads Raw Data (‘output_data_with_regime.csv’)
    - Apply Datetime Rounding - rounding seconds (output_data_with_regime)
      '''
        if seconds >= 30, increases minute by 1;
        Set seconds and microseconds to 0
        Return formatted string
      '''
    - Handling Errors
      '''
       📌 return original string and print a warning for invalid & missing values cases
       📌 use try-except flow statement in handling exceptions
      '''
    - Save Cleaned Data Outputs (‘cleaned_cryptoquant_data.csv’)
    
#### Part 2 - Datetime Alignment and Merging
    - Parse Datetime Strings
      '''
       📌 Convert datetime string columns to datetime objects
      '''
    - Apply Datetime Rounding (BTCUSD)
      '''
       📌 Align Timezones to UTC
      '''
       📌 Ensures both datasets’ key datatime columns are timezone-aware and set to UTC, preventing mismatches during merging
      '''
     - Merge Dataframes
      '''
       📌 Keep only matching datetime
       📌 Add suffixes to distinguish overlapping column names
      '''
     - Save Merged Data (merged_crypto_btcusd_data.csv)
     
#### Part 3: Final Cleaning, Validation, and Structuring
     - Remove Redundant Columns (‘df.drop’)
     - Delete intermediate columns created during parsing and merging
     - Standardize Column Names for clarity(‘df.rename’)
     - Consolidate Timestamps
       '''
        📌 Compares auxiliary timestamp columns (‘start_time, ‘Timestamp’) if present.
        📌  Potentially drops ‘Timestamp’ if deemed redundant based on average difference from ‘start_time
       '''
     - Ensure Data Types
      ''' 
        📌  Confirms final ‘datetime’ column is a datetime object
        📌  Rounds numerical column (OHLCV, flow metrics) to standard decimal places
      ''' 
     - Validate Data Intergrity
      '''
       📌 Checks for & handles potential issues by:
        👉Remove Duplicates (‘df.duplicated’, df.drop_duplicates’)
        👉Check for Missing Values (‘df.isnull().sum’) and prepare for interpolation.
        👉Check Internal Consistency
      '''
     - Structure Data (‘df.sort_values’, column reordering)
      '''
       📌 Sorts dataset chronologically by the main ‘datetime’ column
      '''
📌Save Final Data (‘cleaned_bitcoin_data.csv)

6. HMM NLP Signals Integration
7. Features Engineering
8. Weightage Control and Application
9. Backtesting and ForwardTesting
 - Parameter tuning for optimal number of regimes

  
