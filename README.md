# Sales Conversation Flow Prediction System

This project implements a machine learning pipeline to analyze and improve sales conversations, providing actionable insights to optimize interactions. 
It predicts the next likely **conversation stage**, **response type**, and **conversation direction** using:
* **Embeddings**: Transformer-based models (eg. Sentence-BERT) generate robust conversation representations.
* **Clustering:** Conversations are grouped into meaningful patterns using PCA and KMeans.
* **Sequence modeling:** LSTMs and Markov models capture temporal and sequential dependencies to predict conversational dynamics.

**Key features**
* **Production-ready principles:** Real-time APIs, model versioning and registry, and robust error handling to ensure reliable deployment.
* **Monitoring and retraining:** Automated evaluation scripts, drift detection, and retraining workflows to maintain model performance.
* **Scalability considerations:** Model caching to reduce latency and rate limiting to manage high request loads and prevent abuse during peak usage.

By tackling real-world engineering challenges like conversation embeddings and predictive analytics, 
this project demonstrates skills in **end-to-end machine learning system design, training, evaluation, deployment,** and **monitoring**.


## Project Architecture

1. **Data Preprocessing**
    * Generates embeddings for conversation snippets using Sentence-BERT.
    * Enriches data with metadata (eg. speaker roles, turn numbers) for context.
    * Prepares labeled datasets and sequences for training.
2. **Model Training & Retraining**
    * **LSTM Models:** Predict the next response type and conversation stage.
    * **Markov Models:** Predict conversation direction based on historical cluster transitions.
    * Models are versioned and saved for traceability and reproducibility.
3. **Prediction API**
    * Real-time predictions served via Flask endpoints:
      * **Response Type:** Predicts next likely sales responses.
      * **Conversation Stage:** Identifies current and next conversation stages.
      * **Sales Suggestions:** Recommends actions based on predicted stages and responses.
      * **Conversation Direction:** Predicts conversational flow using Markov models.
4. **Monitoring & Drift Detection**
    * Periodic evaluations track model performance.
    * Drift detection triggers automated retraining workflows to ensure models stay up-to-date.
5. **Scalable & Resilient Design**
    * Caches models to improve response times.
    * Implements input validation, rate limiting, and robust error handling.


## Workflow Overview

**High-Level Steps**
1. **Dataset Preparation:** Process and label raw sales conversation data.
2. **Embedding Generation:** Use Sentence-BERT for dense vector representations.
3. **Clustering:** Group embeddings into patterns with PCA and KMeans.
4. **Model Training:** Train LSTMs for response type and conversation stage prediction.
5. **Deployment:** Serve models via APIs for real-time predictions.
6. **Monitoring & Retraining:** Detect drift and retrain models automatically.

**Core Components**
1. **Preprocessing (`preprocess.py`):** Processes raw conversation data, generates embeddings, and prepares datasets for training.
2. **Training (`train.py`):** Trains LSTM models and saves versioned models with metrics.
3. **Markov Modeling (`markov_model.py`):** Uses embeddings and clusters to predict conversation flow.
4. **API (`app.py`):** Serves real-time predictions through endpoints.
5. **Retraining (`retrain.py`):** Detects drift, retrains models, and updates the registry.
6. **Evaluation (`evaluate.py`):** Measures model performance and logs results.
7. **Monitoring (`periodic_evaluation.py`):** Periodically evaluates models for drift and logs reports.


## APIs

1. **Predict Prospect Response**
    * **Endpoint:** `/predict-prospect-response`
    * **Description:** Predicts the next likely sales response type with probabilities.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "TODO"},
          {"speaker": "Salesman", "response": "TODO"},
          {"speaker": "Customer", "response": "TODO"}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "top_prediction": "Objection",
        "response_type_probabilities": {
          "Objection": 0.71,
          "Question": 0.13,
          "Explanation": 0.06
        }
      }
      ```
      ***Note:** Additional response types omitted for brevity.*

2. **Predict Next Conversation Stage**
    * **Endpoint:** `/predict-next-conversation-stage`
    * **Description:** Predicts the next likely conversation stage with probabilities.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "TODO"},
          {"speaker": "Salesman", "response": "TODO"}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "top_prediction": "Product Discussion",
        "conversation_stage_probabilities": {
          "Introduction": 0.09,
          "Information Gathering": 0.16,
          "Product Discussion": 0.58
          "Objection Handling": 0.04,
          "Closing/Call to Action": 0.05,
          "Other": 0.11,
        }
      }
      ```

3. **Suggest Sales Response**
    * **Endpoint:** `/suggest-sales-response`
    * **Description:** Recommends sales actions based on response and conversation stage predictions.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "TODO"},
          {"speaker": "Salesman", "response": "TODO"}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "suggested_sales_response": "Reassure them with concrete examples of similar customer successes.",
        "top_response_type": "Objection",
        "top_conversation_stage": "Product Discussion",
        "response_type_probabilities": {
          "Objection": 0.71,
          "Question": 0.13,
          "Explanation": 0.06
        },
        "conversation_stage_probabilities": {
          "Introduction": 0.09,
          "Information Gathering": 0.16,
          "Product Discussion": 0.58
          "Objection Handling": 0.04,
          "Closing/Call to Action": 0.05,
          "Other": 0.11,
        }
      }
      ```
      ***Note:** Additional response types omitted for brevity.*

4. **Predict Conversation Direction**
    * **Endpoint:** `/predict-conversation-direction`
    * **Description:** Predicts the next conversational cluster using Markov models.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "TODO"},
          {"speaker": "Salesman", "response": "TODO"}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "current_state": [3, 2],
        "predicted_next_cluster": 5,
        "transition_probabilities": {
          "0": 0.26,
          "1": 0.12,
          "2": 0.13,
          "3": 0.49
        }
      }
      ```


### **Other Output Examples**

**Drift Detection**
1. **Periodic Evaluation Output:**
* **Example Data Drift Output**
    ```plaintext
    Data Drift Report
    - turn - Mean difference: 0.0155
    - speaker - Categorical distribution difference: 0.0000
    - text - Categorical distribution difference: 0.0366
    - sentiment_score - Mean difference: 0.0001
    - sentiment_label - Categorical distribution difference: 0.0067
    - conversation_stage - Categorical distribution difference: 0.0174
    - response_type - Categorical distribution difference: 0.0274
    - cluster - Mean difference: 0.0006
    No significant drift detected for conversation_stage. No retraining required.
    ```
* **Example Performance Metrics Output**
    ```plaintext
    Performance Metrics
    - Accuracy below threshold: 0.5710
    - Precision below threshold: 0.5621
    - Recall below threshold: 0.5710
    - F1_score below threshold: 0.5598
    Drift report updated for conversation_stage.
    ```

2. **Retraining Output:**
    ```plaintext
    Drift detected for conversation_stage. Triggering model retraining.
    Starting model training for conversation_stage...
    Model training completed.
    Retrained Model Metrics for conversation_stage - {'accuracy': 0.745282, 'precision': 0.689314, 'recall': 0.689282, 'f1_score': 0.689653}
    Model for conversation_stage saved to ../models/lstm_models/conversation_stage_model_20241127-233405.h5
    Updated model registry for conversation_stage with version 20241127-233405
    Versioned model for conversation_stage saved successfully.
    ```


## Technologies Used
* **Machine Learning:** Sentence-BERT, LSTMs, Markov models, PCA, KMeans.
* **Backend:** Flask APIs with error handling, rate limiting, and model caching.
* **Experiment Tracking:** MLflow for logging metrics and version control.
* **Deployment:** Docker for containerization.
* **Monitoring:** Drift detection and retraining workflows.


## Future Enhancements

1. **Improved Label Generation**
    * Refine current labels for better prediction accuracy.
    * Add a label for outcomes like "Demo Booked," "Follow-Up Call Needed," and "Prospect Lost" to predict the likelihood of achieving favorable results.
      * In addition: Suggest next-best actions to maximize the likelihood of desired results.
2. **Scalability**
    * Asynchronous predictions for real-time performance.
    * Cloud deployment with load balancing.
3. **Advanced Decision Optimization**
    * Explore reinforcement learning to optimize conversational strategies for sales outcomes such as booking demos.


## Acknowledgements

This project is inspired by real-world challenges in sales optimization and aligns with industry standards for building scalable machine learning systems.
