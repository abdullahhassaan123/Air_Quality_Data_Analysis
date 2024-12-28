# Air_Quality_Data_Analysis
# Capstone Project: Indian Air Quality Data Analysis and Prediction



Following is the link for the dataset: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

The capstone project titled "Indian Air Quality Data Analysis and Prediction" explores India's vast and varied air quality data. With air pollution being a significant public health concern, this project aims to analyze, predict, and identify key trends in air quality across different regions of the country. The project incorporates various machine learning techniques, statistical analyses, and software tools to achieve its objectives.

This project consists of four main files:


# analysis.ipynb: A Jupyter notebook dedicated to data analysis.


# linear_regressor_new.ipynb: A Jupyter notebook implementing linear regression for predictive modeling.


# deployment.py: A Streamlit application for user interaction and expected season-based air quality predictions. (we are not taking into consideration year frames, we are just analysing the trend and anomalies for a 12 months variation in air quality but taking the data of nearly 25 years).
For running the app cmd: streamlit run app/deployment.py

# K-Clustering.ipynb: A notebook focusing on clustering the data using K-means and DBSCAN algorithms.



# 1. Data Analysis (analysis.ipynb)

The first phase of the project involves thorough data exploration and analysis. This step is crucial for understanding the underlying trends and patterns in the data. The "analysis.ipynb" file contains the following components:

1.1 Too many outliers (graph not effective to remove/study outliers)

Outlier Detection: Identified and flagged extreme values for further investigation.

1.2 Data Visualization

Graphs and Plots:

Line charts to observe trends over time.

Bar charts for state-wise and location-wise comparisons.

Heatmap:

A heatmap was generated to show correlations between different pollutants and meteorological variables. This visualization highlighted strong relationships between certain pollutant levels, providing insights into the data's interdependencies.




# 2. Predictive Modeling (linear_regressor_new.ipynb)

In this stage, the project focuses on building a predictive model to estimate RSPM (Reduced Suspended Particulate Matter) concentrations.

2.1 Data Preprocessing

Label Encoding: Transformed categorical variables such as "state," "location," and "area type" into numeric formats suitable for modeling.

Handling Missing Data: Imputed missing values using statistical methods.

2.2 Model Training

Implemented a linear regression model to predict RSPM concentrations based on inputs like state, location, area type, and time of the year.

2.3 Model Performance

The model achieved an R-squared value of 0.30, indicating that 30% of the variability in RSPM levels could be explained by the model. While this value may seem modest, it aligns with the challenges of working with large, dispersed, and noisy data.

Further improvements were attempted by feature engineering; however, the inherent variability of the data limited the R-squared value.



# 3. Application Deployment (deployment.py)

The project includes a Streamlit application for real-time predictions of air quality. The app, "deployment.py," allows users to input parameters and receive predictions.

3.1 User Inputs

State: The state for which the prediction is to be made.

Location: Specific location within the state.

Area Type: Rural or urban classification.

Time of Year: Month and date for which the prediction is desired.

3.2 Prediction via Streamlit App

Based on the user inputs, the app predicts the expected RSPM concentration using the trained linear regression model.

The app categorizes air quality into "Good," "Moderate," "Poor," or "Hazardous" based on the predicted RSPM value, providing actionable insights for stakeholders.



# 4. Clustering and Anomaly Detection (K-Clustering.ipynb)

The "K-Clustering.ipynb" notebook focuses on clustering and anomaly detection to identify patterns and irregularities in the data.

4.1 K-Means Clustering

The model clusters the data based on air quality attributes like SO2, NO2, RSPM, and SPM, state, type, month, date.

Optimal cluster count was determined using the silhouette score, ensuring meaningful separation of clusters.

The clustering results revealed distinct patterns, aiding in categorizing regions with similar pollution levels.

4.2 Anomaly Detection

Implemented DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to detect anomalies in the data.

While DBSCAN effectively identified outliers, K-means showed superior performance in terms of silhouette score.

4.3 Anomaly Insights

The notebook calculates the number of anomalies for each state, providing actionable insights for policymakers. For instance, states with high anomaly counts may indicate data collection issues or unique environmental challenges.



# Conclusion and Future Work

This capstone project demonstrates the potential of data science in addressing critical environmental challenges. By analyzing and predicting air quality, the project offers valuable insights for improving air quality management in India.


# Achievements

Comprehensive analysis of air quality data.

Development of a predictive model for RSPM levels.

Deployment of a user-friendly application for real-time predictions.

Clustering and anomaly detection to uncover hidden patterns and irregularities.



# Future Directions

Incorporating advanced models like Random Forests or Neural Networks to improve prediction accuracy.

Expanding the app to include more pollutants and factors affecting air quality.

Enhancing anomaly detection methods for better reliability and accuracy.

Collaborating with government and environmental agencies to implement data-driven policies for air quality improvement.

This project lays a strong foundation for leveraging data science in tackling environmental challenges, emphasizing the importance of analytics in informed decision-making. By combining robust analysis, predictive modeling, and practical deployment, the "Indian Air Quality Data Analysis and Prediction" project contributes meaningfully to the discourse on air quality management in India and likewise in Pakistan as well.

