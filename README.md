# Sales Conversation Flow Prediction System

This project implements a machine learning pipeline to analyze and improve sales conversations, providing actionable insights to optimize 
interactions. It predicts the next likely **response type**, **conversation stage**, and **conversation direction** while **recommending 
tailored sales responses** and **forecasting conversation dynamics** to drive better outcomes.

This is achieved using:
* **Embeddings**: Transformer-based models (eg. Sentence-BERT) generate robust conversation representations.
* **Clustering:** Conversations are grouped into meaningful patterns using PCA and KMeans.
* **Sequence modeling:** LSTMs and Markov models capture temporal and sequential dependencies to predict conversational dynamics.

## Key Features

The following features demonstrate the project’s production-ready capabilities and focus on machine learning system design:
* **Production-ready principles:**
   * Real-time **APIs** to serve predictions in low-latency scenarios.
   * Robust **model versioning and registry** for seamless deployment and rollback.
   * End-to-end **error handling** and input validation to ensure reliability.
* **Monitoring and retraining:**
   * Automated **evaluation scripts** continuously monitor performance metrics.
   * **Drift detection** identifies when retraining is required.
   * Fully automated **retraining workflows** integrate new models into production without manual intervention.
* **Scalability considerations:**
   * **Model caching** minimizes inference latency.
   * **Rate limiting** ensures resilience during high traffic, managing high request loads and preventing abuse during peak usage.

By addressing real-world engineering challenges, this project demonstrates skills in **end-to-end machine learning system design**:
* **Dataset preparation** and **data pipeline engineering**.
* **Training, evaluation, and deployment of machine learning models**.
* **Automation of performance monitoring and retraining workflows**.
* **Designing scalable and resilient ML systems** for production environments.


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

## Dataset

The dataset used in this project is sourced from the [Sales Conversations dataset](https://huggingface.co/datasets/goendalf666/sales-conversations) on Hugging Face. It contains annotated sales dialogues, enabling exploration of conversation stages, response types, and conversational dynamics.

## Exploratory Analysis and Prototyping

This project includes Jupyter notebooks for exploratory data analysis and prototyping, providing insights that guided the production pipeline design:
* `EDA.ipynb`: Examines the raw sales conversation dataset to uncover patterns, feature distributions, and preprocessing requirements.
* `label_generation.ipynb`: Creates and verifies conversation labels for downstream tasks.
* `EDA_labels.ipynb`: Explores the dataset while validating the quality, balance, and correctness of generated labels (eg. response_type, conversation_stage).
* `cluster_analysis.ipynb`: Evaluates conversation embeddings to select the optimal clustering approach. It explores embedding models, uses PCA for dimensionality reduction, and compares clustering metrics, identifying KMeans with 4 clusters as the most effective. Includes visualizations aimed to highlight meaningful patterns.

These notebooks bridge the gap between exploratory work and the deployment-ready pipeline, ensuring a strong foundation for the system.


## APIs

1. **Predict Prospect Response**
    * **Endpoint:** `/predict-prospect-response`
    * **Description:** Predicts the next likely sales response type with probabilities.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "I think the price is a bit too high."},
          {"speaker": "Salesman", "response": "Many customers felt that way initially, but they found significant value in the product after using it."},
          {"speaker": "Customer", "response": "That makes sense. Can you share more details about how it works?"},
          {"speaker": "Salesman", "response": "Of course! Here's a quick overview of how our product can solve your problems effectively. And so on..."}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "top_prediction": "Question",
        "response_type_probabilities": {
          "Question": 0.64,
          "Agreement": 0.18,
          "Objection": 0.12,
          "Explanation": 0.06
        }
      }
      ```
      ***Note:** Additional response types omitted for brevity.*
      * Based on the history, the system predicts that the customer is most likely to ask another question (64% probability), as they might want further clarification or more details.

2. **Predict Next Conversation Stage**
    * **Endpoint:** `/predict-next-conversation-stage`
    * **Description:** Predicts the next likely conversation stage with probabilities.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "Can you tell me more about how this works?"},
          {"speaker": "Salesman", "response": "Absolutely! Our platform makes the sales process easier by automating follow-ups and giving you detailed insights about your prospects."}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "top_prediction": "Product Discussion",
        "conversation_stage_probabilities": {
          "Introduction": 0.09,
          "Information Gathering": 0.15,
          "Product Discussion": 0.58,
          "Objection Handling": 0.03,
          "Closing/Call to Action": 0.04,
          "Other": 0.11,
        }
      }
      ```
      * Based on the history, the system predicts that the next stage of the conversation is most likely to focus on Product Discussion (58% probability), where the salesman and customer dive deeper into the product’s details and features.

3. **Suggest Sales Response**
    * **Endpoint:** `/suggest-sales-response`
    * **Description:** Recommends sales actions based on response and conversation stage predictions.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "I'm not sure this product is what we need."},
          {"speaker": "Salesman", "response": "I understand your concerns. Could you share more about what you're looking for in a solution?"}
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
      * Based on the conversation history, the system predicts that the customer’s next response is likely to be an Objection (71% probability), indicating hesitancy or concern. Given that the conversation is currently in the Product Discussion stage (58% probability), the suggested sales response is to “Reassure them with concrete examples of similar customer successes.” This approach helps address the objection while keeping the conversation productive.

4. **Predict Conversation Direction**
    * **Endpoint:** `/predict-conversation-direction`
    * **Description:** Predicts the next conversational cluster using Markov models.
    * **Method:** POST
    * **Input:**
      ```json
      {
        "history": [
          {"speaker": "Customer", "response": "Can you explain how the pricing works?"},
          {"speaker": "Salesman", "response": "Absolutely! Our pricing is based on usage tiers, so smaller teams can start at a lower cost, and scaling up is seamless."}
        ]
      }
      ```
    * **Output:**
      ```json
      {
        "current_state": [3, 2],
        "predicted_next_cluster": 3,
        "transition_probabilities": {
          "0": 0.26,
          "1": 0.12,
          "2": 0.13,
          "3": 0.49
        }
      }
      ```
      * The system identifies the current conversation flow using clusters: 3 for the customer’s query and 2 for the salesperson’s response. Based on these clusters, the Markov model predicts a transition to Cluster 3 (49% probability), reflecting the most likely direction of the conversation.
      * While the exact meaning of these clusters depends on the underlying embedding and clustering strategy, this transition can guide further analysis or actionable suggestions for the sales process.


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
* **Experiment Tracking:** MLflow for logging metrics, model versioning, and performance tracking.
* **Model Registry:** Centralized YAML-based model registry for version control and traceability.
* **Deployment:** Docker for containerization.
* **Monitoring:** Drift detection, retraining workflows, and automated integration of updated models.


## Future Enhancements

1. **Improved Label Generation**
    * Refine current labels for better prediction accuracy.
    * Introduce outcome-oriented labels such as “Demo Booked,” “Follow-Up Call Needed,” and “Prospect Lost” to enable predicting the likelihood of achieving favorable results based on conversation history (eg. the first half of a conversation).
      * Recommend next-best actions to maximize the likelihood of desired outcomes, such as booking a demo or making a sale.
2. **Scalability**
    * Asynchronous predictions for real-time performance.
    * Cloud deployment with load balancing.
3. **Advanced Decision Optimization**
    * Explore reinforcement learning to optimize conversational strategies for sales outcomes such as booking demos.


## Acknowledgements

This project is inspired by real-world challenges in sales optimization and aligns with industry standards for building scalable machine learning systems.
