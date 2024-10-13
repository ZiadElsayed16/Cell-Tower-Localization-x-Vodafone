import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from data_utils import load_data, clean_data, create_preprocessing_pipeline

class RandomForestModel:
    """
    Class for creating and managing a Random Forest model pipeline for predicting 
    geographical coordinates (Latitude and Longitude).

    Attributes:
    data_path (str): Path to the dataset for training.
    model_path (str): Path to the saved model for inference.
    df_reg (pd.DataFrame): DataFrame holding the loaded dataset.
    model (RandomForestRegressor): Random Forest regressor model.
    pipeline (Pipeline): Pipeline for preprocessing and modeling.
    """

    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the RandomForestModel class.

        Args:
        data_path (str): Path to the dataset for training (default is None).
        model_path (str): Path to the pre-trained model for inference (default is None).
        """
        self.data_path = data_path
        self.model_path = model_path
        self.df_reg = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pipeline = None

    def load_data(self):
        """
        Load the dataset from the specified file path into a DataFrame.
        """
        if self.data_path:
            self.df_reg = load_data(self.data_path)

    def clean_data(self):
        """
        Clean the dataset by applying data cleaning transformations.
        """
        
        self.df_reg = clean_data(self.df_reg)

    def split_data(self, target_columns=['Latitude', 'Longitude'], test_size=0.2):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        - target_columns (list): A list of column names to be used as target variables.
                                Default is ['Latitude', 'Longitude'].
        - test_size (float): Proportion of the dataset to include in the test split. 
                            Must be between 0.0 and 1.0. Default is 0.2 (20%).

        Returns:
        - None: The function modifies the instance variables self.X_train, self.X_test, 
                self.y_train, and self.y_test with the training and testing data.

        Example:
        >>> rf_model.split_data(target_columns=['Latitude', 'Longitude'], test_size=0.3)
        """
        
        X = self.df_reg.drop(columns=target_columns)
        y = self.df_reg[target_columns]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)


    def create_pipeline(self):
        """
        Create a data preprocessing and modeling pipeline.
        """
        numerical_features = ['client_latitude', 'client_longitude', 'dbm_a', 'rsrp_a', 'rsrq_a', 'Site_ID', 'download_kbps']
        categorical_features = ['client_city', 'brand', 'Band']

        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)

        # Create pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

    def train_model(self):
        """
        Train the model using the training data.
        """
        self.pipeline.fit(self.X_train, self.y_train)

    def save_model(self, filepath):
        """
        Save the trained pipeline to a file.

        Args:
        filepath (str): The path where the model will be saved.
        """
        joblib.dump(self.pipeline, filepath)

    def evaluate_model(self):
        """
        Evaluate the model performance and display plots comparing predicted and actual values.
        """
        train_score = self.pipeline.score(self.X_train, self.y_train)
        test_score = self.pipeline.score(self.X_test, self.y_test)
        print(f'Model R^2 train score: {train_score:.4f}')
        print(f'Model R^2 test score: {test_score:.4f}')
        
        # Plot predictions vs actual values
        y_pred = self.pipeline.predict(self.X_test)
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test['Latitude'], y_pred[:, 0], alpha=0.3)
        plt.xlabel('Actual Latitude')
        plt.ylabel('Predicted Latitude')
        plt.title('Latitude Predictions vs Actual')

        plt.subplot(1, 2, 2)
        plt.scatter(self.y_test['Longitude'], y_pred[:, 1], alpha=0.3)
        plt.xlabel('Actual Longitude')
        plt.ylabel('Predicted Longitude')
        plt.title('Longitude Predictions vs Actual')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./Plots/Models/RF Longitude Predictions vs Actual.png')
        plt.close()

    def predict(self, data_path):
        """
        Perform inference on new data and return predictions.

        Args:
        data_path (str): Path to the new dataset for inference.

        Returns:
        np.ndarray: Predictions for the given dataset.
        """
        X_new = pd.read_excel(data_path)
        predictions = self.pipeline.predict(X_new)
        
        # Save predictions to Excel
        predictions_df = pd.DataFrame(predictions, columns=['Latitude', 'Longitude'])
        predictions_df.to_excel('./predictions/RF-pred.xlsx', index=False)
        print(predictions)
        return predictions

    def load_trained_model(self):
        """
        Load a pre-trained model from the specified file path.
        """
        if self.model_path:
            self.pipeline = joblib.load(self.model_path)

# User interaction script
if __name__ == "__main__":
    mode = input("Do you want to train the model or use inference mode? (train/inference): ").strip().lower()
    
    if mode == "train":
        data_path = input("Please provide the path to the training dataset: ").strip()
        model = RandomForestModel(data_path=data_path)
        model.load_data()
        model.clean_data()
        model.split_data()
        model.create_pipeline()
        model.train_model()
        model.save_model('./saved-models/random_forest_pipeline.pkl')
        model.evaluate_model()
        print("Model trained and saved successfully.")

    elif mode == "inference":
        model_path = input("Please provide the path to the pre-trained model: ").strip()
        data_path = input("Please provide the path to the new dataset for predictions: ").strip()
        model = RandomForestModel(model_path=model_path)
        model.load_trained_model()
        predictions = model.predict(data_path)
        print("Inference completed. Predictions are displayed above.")
    
    else:
        print("Invalid option. Please choose either 'train' or 'inference'.")
