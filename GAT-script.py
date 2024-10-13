from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
import numpy as np
from data_utils import load_data, clean_data, create_preprocessing_pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GATRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initializes the GATRegressor model.

        Parameters:
        - in_channels (int): Number of input features.
        - hidden_channels (int): Number of hidden features.
        - out_channels (int): Number of output features (e.g., Latitude and Longitude).
        """
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.fc = nn.Linear(hidden_channels * 4, out_channels)

    def forward(self, data):
        """
        Forward pass through the model.

        Parameters:
        - data: A Data object containing node features and edge indices.

        Returns:
        - x: Predicted output for the input data.
        """
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x


class GATPipeline:
    def __init__(self, data_path):
        """
        Initializes the GATPipeline.

        Parameters:
        - data_path (str): Path to the dataset.
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads the dataset from the specified path."""
        try:
            self.df = load_data(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")

    def clean_data(self):
        """Cleans the loaded dataset."""
        self.df = clean_data(self.df)

    def split_data(self):
        """
        Splits the dataset into training, validation, and testing sets.

        Returns:
        - X_train, X_val, X_test: Feature sets for training, validation, and testing.
        - y_train, y_val, y_test: Target sets for training, validation, and testing.
        """
        target_columns = ['Latitude', 'Longitude']
        X = self.df.drop(columns=target_columns)
        y = self.df[target_columns]

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def compute_edges_knn(self, X, n_neighbors=10):
        """
        Computes edges using k-nearest neighbors.

        Parameters:
        - X: DataFrame containing feature data.
        - n_neighbors (int): Number of neighbors to consider for edge creation.

        Returns:
        - edges (numpy.ndarray): Array of edges for the graph.
        """
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(X[['client_latitude', 'client_longitude', 'Site_ID']])

        distances, indices = knn.kneighbors(X[['client_latitude', 'client_longitude', 'Site_ID']])
        edges = []
        for i in range(X.shape[0]):
            for j in indices[i]:
                if i != j:  # Prevent self-loops
                    edges.append([i, j])

        return np.array(edges).T

    def create_graph_data(self, X, edges, y=None):
        """
        Creates a graph data object for GAT training or inference.

        Parameters:
        - X: Feature data.
        - edges: Array of edges for the graph.
        - y: Target values (optional).

        Returns:
        - data: A Data object suitable for GAT processing.
        """
        if y is not None:
            y = torch.tensor(y, dtype=torch.float).to(device)
            data = Data(x=torch.tensor(X, dtype=torch.float).to(device),
                        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
                        y=y)  # Only include y if it's not None
        else:
            data = Data(x=torch.tensor(X, dtype=torch.float).to(device),
                        edge_index=torch.tensor(edges, dtype=torch.long).to(device))

        return data

    def create_pipeline(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Creates a preprocessing pipeline and prepares data for GAT training.

        Parameters:
        - X_train, X_val, X_test: Feature sets for training, validation, and testing.
        - y_train, y_val, y_test: Target sets for training, validation, and testing.

        Returns:
        - model: The GATRegressor model.
        - optimizer: The optimizer for training.
        - criterion: The loss function.
        - train_data, val_data, test_data: Graph data objects for training, validation, and testing.
        - X_test_scaled: Scaled test features.
        - y_test_np: NumPy array of test targets.
        """
        numerical_features = ['client_latitude', 'client_longitude', 'dbm_a', 'rsrp_a', 'rsrq_a', 'Site_ID', 'download_kbps']
        categorical_features = ['client_city', 'brand', 'Band']
        
        preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        X_test_scaled = preprocessor.transform(X_test)

        joblib.dump(preprocessor, './saved-models/GAT_preprocessor_pipeline.pkl')

        if not isinstance(X_train_scaled, np.ndarray):
            X_train_scaled = X_train_scaled.toarray()
            X_val_scaled = X_val_scaled.toarray()
            X_test_scaled = X_test_scaled.toarray()

        edges_train = self.compute_edges_knn(X_train, n_neighbors=5)
        edges_val = self.compute_edges_knn(X_val, n_neighbors=5)
        edges_test = self.compute_edges_knn(X_test, n_neighbors=5)

        y_train_np = y_train.to_numpy()
        y_val_np = y_val.to_numpy()
        y_test_np = y_test.to_numpy()

        train_data = self.create_graph_data(X_train_scaled, edges_train, y_train_np)
        val_data = self.create_graph_data(X_val_scaled, edges_val, y_val_np)
        test_data = self.create_graph_data(X_test_scaled, edges_test, y_test_np)

        model = GATRegressor(in_channels=X_train_scaled.shape[1], hidden_channels=64, out_channels=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        return model, optimizer, criterion, train_data, val_data, test_data, X_test_scaled, y_test_np

    def run_training(self, model, optimizer, criterion, train_data, val_data, test_data):
        """
        Runs the training loop for the GAT model.

        Args:
            model (torch.nn.Module): The GAT model to train.
            optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
            criterion (torch.nn.Module): The loss function to minimize during training.
            train_data (Data): The training data for the GAT.
            val_data (Data): The validation data for the GAT.
            test_data (Data): The test data for evaluation.

        Returns:
            float: R² score of the model on the test dataset.

        This function performs the following steps:
        - Trains the model for a specified number of epochs, tracking the training and validation loss.
        - Implements early stopping based on validation loss to prevent overfitting.
        - After training, evaluates the model on the test dataset and computes the R² score.
        - Visualizes the predictions versus actual values for latitude and longitude using scatter plots.
        - Displays a diagonal line representing the ideal case where predicted values match actual values.

        """
        patience = 20
        best_val_loss = float('inf')
        counter = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        model_path = "./saved-models/GAT_model.pt"

        for epoch in range(4000):
            model.train()
            optimizer.zero_grad()
            out = model(train_data)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()

            model.eval()
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y)

            # Calculate R² scores
            train_r2 = r2_score(train_data.y.cpu().numpy(), out.cpu().detach().numpy())
            val_r2 = r2_score(val_data.y.cpu().numpy(), val_out.cpu().detach().numpy())

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                    f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping!")
                break

        # Load the best model and evaluate on test data
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_out = model(test_data)

        # Compute R² score
        test_r2 = r2_score(test_data.y.cpu().numpy(), test_out.cpu().detach().numpy())

        # Visualize Predictions vs Actual Values
        y_actual = test_data.y.cpu().numpy()
        y_predicted = test_out.cpu().detach().numpy()

        # Create subplots for Latitude and Longitude
        plt.figure(figsize=(14, 6))

        # Plot for Latitude
        plt.subplot(1, 2, 1)
        plt.scatter(y_actual[:, 0], y_predicted[:, 0], alpha=0.3)
        plt.xlabel('Actual Latitude')
        plt.ylabel('Predicted Latitude')
        plt.title('Latitude Predictions vs Actual')

        # Plot for Longitude
        plt.subplot(1, 2, 2)
        plt.scatter(y_actual[:, 1], y_predicted[:, 1], alpha=0.3)
        plt.xlabel('Actual Longitude')
        plt.ylabel('Predicted Longitude')
        plt.title('Longitude Predictions vs Actual')
        

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./Plots/Models/GAT Longitude Predictions vs Actual.png')
        plt.close()

        return test_r2

    def run_inference(self, model_path, preprocessor_path, X_new):
        """
        Runs inference on new data using the trained model.

        Parameters:
        - model_path (str): Path to the saved model.
        - preprocessor_path (str): Path to the saved preprocessing pipeline.
        - X_new: DataFrame containing new input data for predictions.

        Returns:
        - predictions: Numpy array of predicted latitude and longitude values.
        """
        lat_lon_site = X_new[['client_latitude', 'client_longitude', 'Site_ID']]
        preprocessor = joblib.load(preprocessor_path)
        X_new_scaled = preprocessor.transform(X_new)

        # Convert to dense array if necessary
        if hasattr(X_new_scaled, 'toarray'):
            X_new_scaled = X_new_scaled.toarray()
            
        edges = self.compute_edges_knn(lat_lon_site)
        # Pass None for y as it's not needed during inference
        graph_data = self.create_graph_data(X_new_scaled, edges, None)

        model = GATRegressor(in_channels=X_new_scaled.shape[1], hidden_channels=64, out_channels=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            predictions = model(graph_data)

        # Save predictions to Excel
        predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=['Latitude', 'Longitude'])
        predictions_df.to_excel('./predictions/GAT_predictions.xlsx', index=False)

        return predictions.cpu().numpy()


# Main execution
def main():
    mode = input("Enter mode (train or inference): ").strip().lower()
    data_path = input("Enter the data path: ")

    pipeline = GATPipeline(data_path)
    pipeline.load_data()

    if mode == "train":
        pipeline.clean_data()
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data()
        model, optimizer, criterion, train_data, val_data, test_data, X_test_scaled, y_test_np = pipeline.create_pipeline(X_train, X_val, X_test, y_train, y_val, y_test)
        test_r2 = pipeline.run_training(model, optimizer, criterion, train_data, val_data, test_data)
        print(f"Final R-squared on the test set: {test_r2}")

    elif mode == "inference":
        model_path = input("Enter the model path: ")
        preprocessor_path = input("Enter the preprocessor path: ")
        pipeline.clean_data()  # Only clean the data
        X_new = pipeline.df  # Use the cleaned data
        predictions = pipeline.run_inference(model_path, preprocessor_path, X_new)
        print("Predictions:", predictions)

    else:
        print("Invalid mode entered. Please choose either 'train' or 'inference'.")

if __name__ == "__main__":
    main()
