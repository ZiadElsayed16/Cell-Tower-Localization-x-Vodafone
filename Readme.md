# Mobile Network Tower Location Identification Using ML/AI

## Project Overview
This project aims to build a predictive model to determine the actual locations of mobile towers based on network performance metrics, device information, and geographic data. The model predicts the latitude and longitude coordinates of mobile towers that clients connect to by leveraging features such as signal strength, device type, and geographic attributes.

## Methods and Approaches
   ### ***We used two distinct paradigm to achieve the goal of predicting tower locations:***

1. **Geometric Approaches**:
   - ### Triangulation
        A fundamental concept in fields like physics, geodesy, and telecommunications, used to determine the location of an object by measuring angles or distances from known points, And the concept behind triangulation for estimating cell tower locations relies on using the position and signal strength data from multiple clients to determine the most likely position of a tower.

        **This project implements two triangulation methods for estimating the location of cell towers based on client data:**

        #### ***Group-Based Triangulation***:
        - Groups client data by cell tower ID
        - Calculates weighted average of client latitudes and longitudes
        - Weights derived from signal strength (dbm_a)
        - Provides robust estimation with multiple data points

        #### ***Single-Row Triangulation***:
        - Uses a window of nearby data points for each client row
        - Estimates tower location based on local data
        - Flexible window size for adjusting accuracy
        - Suitable for scenarios with limited data points

    - ### Multilateration
        Multilateration estimates an object's position using known locations of multiple reference points and measured distances or signal travel times, It is commonly applied in geolocation, radar, and telecommunications to determine the location of a signal source.

        **This project implements two multilateration methods for estimating the location of cell towers based on client data:**

        #### ***geographical Multilateration***:
        - Uses client locations and signal strengths
        - Optimizes tower position by minimizing errors between estimated and actual distances
        - Applied individually for each cell tower


        #### ***Least Squares Multilateration***:
        - Estimates distances using Received Signal Reference Power (RSRP)
        - Employs least squares optimization for tower location
        - Groups data by site ID, potentially combining multiple tower information

    ### ***All the Geometric Approaches are evaluated using visualizations (scatter plots, distribution plots) and statistical measures (R² scores). This allows for comprehensive assessment of each approach's effectiveness in various scenarios.***




2. **Machine Learning Methods**:
   - ### Random Forest
        A traditional ensemble learning method that uses multiple decision trees to predict the geographical coordinates (Latitude and Longitude) of mobile towers based on various network performance and device-related features. The pipeline is designed to clean, preprocess, and model the data efficiently, allowing for both training and inference workflows.


    - ### Graph Attention Network (GAT) Approach

        The **Graph Attention Network (GAT)** method is used to predict the latitude and longitude of mobile towers by leveraging spatial relationships between different network elements. This approach is particularly effective for capturing complex interactions in the graph-based data structure where nodes represent clients or tower connections and edges represent k-nearest neighbors.

        #### Key Aspects of the GAT Model:
        - **Node Features**: Input features include network performance metrics such as signal strength (`dbm_a`, `rsrp_a`, `rsrq_a`), geographic attributes (`client_latitude`, `client_longitude`), and device/site identifiers.
        - **Graph Construction**: We compute edges using a **k-nearest neighbors (k-NN)** algorithm, where each node is connected to its nearest neighbors based on spatial features like latitude, longitude, and site ID.
        - **GAT Architecture**: The GAT model consists of two graph attention layers. The first layer applies multiple attention heads to focus on different parts of the neighborhood for each node, followed by an aggregation step. A second GAT layer further refines the node representations. Finally, a fully connected layer predicts the latitude and longitude.
        
        - **Input Layer**: Accepts node features.
        - **Hidden Layer**: Two layers of GAT with multi-head attention.
        - **Output Layer**: Fully connected linear layer for predicting tower latitude and longitude.

        #### Model Training:
        - **Loss Function**: The training process uses Mean Squared Error (MSE) as the loss function to minimize the difference between the predicted and actual coordinates.
        - **Early Stopping**: Early stopping is implemented to prevent overfitting, using validation loss to monitor model performance.
        - **R² Evaluation**: The model is evaluated using the R² score, which measures how well the predictions align with the actual tower locations.
        
        The model is trained over several epochs, and performance is visualized by plotting predicted vs. actual values for latitude and longitude.

        #### Inference:
        The GAT model is also capable of running inference on new data. After scaling the features and computing edges, the model predicts the tower's latitude and longitude for the new input data. The results are saved as an Excel file for further analysis.

You can run the training or inference by using the appropriate mode in the script.


Each of these methods has its own dedicated Python script (`.py` file).

## Project Structure
```plaintext
├── inference-data/                         # Directory for inference datasets
│   └── Test-data.xlsx                      # Example test dataset file
├── training-data/                          # Directory for training datasets
│   └── DEBI_Data_V2.xlsx                   # Example training dataset file
├── Plots/                                  # Directory for the plots of all aproaches
│   └── Models/                             # Directory for the plots of GAT and Random Forest
│   └── Multilateration/                    # Directory for the plots of Multilateration
│   └── Triangulation/                      # Directory for the plots of Triangulation
├── predictions/                            # Directory for all predictions excel files
│   └── RF-pred.xlsx                        # Example predictions file here using Random Forest
├── saved-models/                           # Directory for all saved files to run inference
│   └── GAT_preprocessor_pipeline.pkl       # GAT saved preprocessor 
│   └── GAT_model.pt                        # GAT saved model
│   └── random_forest_pipeline.pkl          # random forest saved pipeline
├── triangulation.py                        # Script for Triangulation approach
├── multilateration.py                      # Script for Multilateration approach
├── RF-script.py                            # Script for Random Forest model
├── GAT-script.py                           # Script for GAT-based model
├── data_utils.py                           # Script for handling data e.g. load, clean, etc.
└── README.md                  # This README file

```

## Data

The dataset used in this project consists of several key features related to mobile network performance and device attributes. The main features include:

- **test_id:** Unique identifier for each network test.
- **test_date:** Date of the test.
- **download_kbps:** Download speed measured in kilobits per second.
- **upload_kbps:** Upload speed measured in kilobits per second.
- **client_city:** The city where the client is located.
- **client_latitude:** Geographic coordinate indicating the client's latitude.
- **client_longitude:** Geographic coordinate indicating the client's longitude.
- **brand:** The brand of the device used for the test.
- **start_cell_id_a:** Starting cell ID of the network connection.
- **dbm_a:** Signal strength measurement in dBm.
- **rsrp_a:** Reference Signal Received Power measurement.
- **rsrq_a:** Reference Signal Received Quality measurement.
- **Technology:** Network technology (e.g., LTE).
- **Band:** Frequency band used for the connection.
- **Latitude:** Actual geographic coordinate of the connected mobile tower (target variable).
- **Longitude:** Actual geographic coordinate of the connected mobile tower (target variable).
- **Site_ID:** Unique identifier for the mobile tower site.


## Installation

1. **Set up the environment**: Create and activate a Python environment using the provided requirements.txt file:
    
    **eg.** using conda

   ```bash
   conda create --name tower-predict-env --file requirements.txt
   conda activate tower-predict-env
   ```
2. **Run the Python scripts**: You can run each method individually by executing the corresponding Python script:

    ```bash
   python Triangulation.py
   python Multilateration.py
   ```
   ### In case of Random Forest and GAT there are special instructions to follow

    - **Random Forest:**

        The user will be free to choose whether to train the model or use the inference mode

        If one choose train, will have to provide the training dataset path

        ```bash
        python RF-script.py
        Do you want to train the model or use inference mode? (train/inference):: train
        Please provide the path to the training dataset: ./training-data/DEBI_Data_V2.xlsx
        ```
        If one choose inference, will have to provide the pre-traind model, inference dataset path

        ```bash
        python RF-script.py
        Do you want to train the model or use inference mode? (train/inference): inference
        Please provide the path to the pre-trained model: ./saved-models/random_forest_pipeline.pkl 
        Please provide the path to the new dataset for predictions: ./inference-data/Test-data.xlsx 
        ```
    
    - **GAT:**

        The user will be free to choose whether to train the model or use the inference mode

        If one choose train, will have to provide the training dataset path

        ```bash
        python GAT-script.py
        Enter mode (train or inference): train
        Enter the data path: ./training-data/DEBI_Data_V2.xlsx
        ```
        If one choose inference, will have to provide the pre-traind model, inference dataset path

        ```bash
        python GAT-script.py
        Enter mode (train or inference): inference
        Enter the data path: ./inference-data/Test-data.xlsx 
        Enter the model path: ./saved-models/GAT_model.pt      
        Enter the preprocessor path: ./saved-models/GAT_preprocessor_pipeline.pkl 
        ```

