import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score
from scipy.optimize import minimize, least_squares
import time
from tqdm import tqdm
from data_utils import load_data, clean_data
tqdm.pandas()

class TriangulationEstimator:
    """
    A class for estimating tower locations using triangulation methods.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing cell tower and client data.
        Tri_Appr (pd.DataFrame): Results of group-based triangulation estimation.
        Tri_Appr_single (pd.Series): Results of single-row triangulation estimation.
    """

    def __init__(self, df):
        """
        Initialize the TriangulationEstimator with input data.

        Args:
            df (pd.DataFrame): The input DataFrame containing cell tower and client data.
        """
        self.df = df
        self.Tri_Appr = None
        self.Tri_Appr_single = None

    def run_all_estimations(self):
        """
        Run all triangulation estimations.

        Side effects:
            Calls run_triangulation method.
        """
        print("Running triangulation...")
        self.run_triangulation()

    def run_triangulation(self):
        """
        Perform triangulation estimations using group-based and single-row methods.

        Side effects:
            Sets self.Tri_Appr and self.Tri_Appr_single with estimation results.
        """
        self.Tri_Appr = self.df.groupby('start_cell_id_a').progress_apply(self.estimate_tower_location)
        self.Tri_Appr_single = self.df.progress_apply(lambda row: self.estimate_tower_location_with_window(self.df, row), axis=1)

    @staticmethod
    def estimate_tower_location(group):
        """
        Estimate tower location for a group of data points.

        Args:
            group (pd.DataFrame): A group of data points for a single cell tower.

        Returns:
            pd.Series: Estimated latitude and longitude for the tower.
        """
        weights = 1 / (np.abs(group['dbm_a']) + 1e-6)
        weighted_latitude = np.average(group['client_latitude'], weights=weights)
        weighted_longitude = np.average(group['client_longitude'], weights=weights)
        return pd.Series({'estimated_tower_latitude': weighted_latitude, 'estimated_tower_longitude': weighted_longitude})

    @classmethod
    def estimate_tower_location_with_window(cls, df, client_row, window_size=20):
        """
        Estimate tower location for a single data point using a window of nearby points.

        Args:
            df (pd.DataFrame): The full DataFrame containing all data points.
            client_row (pd.Series): A single row of client data.
            window_size (int, optional): Number of nearby points to consider. Defaults to 20.

        Returns:
            pd.Series: Estimated latitude and longitude for the tower.
        """
        nearby_clients = df[(df['start_cell_id_a'] == client_row['start_cell_id_a'])].head(window_size)
        return cls.estimate_tower_location(nearby_clients)

class TriangulationResultAnalyzer:
    """
    A class for analyzing and visualizing the results of triangulation estimations.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing cell tower and client data.
        estimator (TriangulationEstimator): The estimator object containing triangulation results.
        comparisons (list): A list of tuples containing comparison data and metadata.
    """

    def __init__(self, df, estimator):
        """
        Initialize the TriangulationResultAnalyzer with input data and estimator results.

        Args:
            df (pd.DataFrame): The input DataFrame containing cell tower and client data.
            estimator (TriangulationEstimator): The estimator object containing triangulation results.
        """
        self.df = df
        self.estimator = estimator
        self.comparisons = None

    def prepare_comparisons(self):
        """
        Prepare comparison data for analysis and visualization.

        Side effects:
            Sets self.comparisons with prepared comparison data.
            Saves comparison data to Excel files.
        """
        long_lat_grouped_by_cell_id = self.df.groupby('start_cell_id_a')[['Latitude', 'Longitude']].mean().reset_index()

        comparison_TriSingle = pd.merge(self.estimator.Tri_Appr_single, self.df[['Site_ID', 'Latitude', 'Longitude', 'start_cell_id_a']], left_index=True, right_index=True, suffixes=('_est', '_true'))
        comparison_Tri = pd.merge(self.estimator.Tri_Appr, long_lat_grouped_by_cell_id, on='start_cell_id_a', suffixes=('_est', '_true'))

        self.comparisons = [
            (comparison_Tri, 'Comparison Tri', 'estimated_tower_latitude', 'estimated_tower_longitude', 'Latitude', 'Longitude'),
            (comparison_TriSingle, 'Comparison TriSingle', 'estimated_tower_latitude', 'estimated_tower_longitude', 'Latitude', 'Longitude')
        ]
        comparison_Tri.to_excel('./predictions/comparison_Tri.xlsx', index=False)
        comparison_TriSingle.to_excel('./predictions/comparison_TriSingle.xlsx', index=False)

    def run_analysis(self):
        """
        Run analysis on prepared comparison data.

        Side effects:
            Calls various analysis and visualization methods.
            Prints R² scores for latitude and longitude estimations.
        """
        for comparison, title, est_lat, est_lon, true_lat, true_lon in self.comparisons:
            self.analyze_comparison(comparison, est_lat, est_lon, true_lat, true_lon, title)
            self.plot_comparison(comparison, est_lat, est_lon, true_lat, true_lon, title)
            self.plot_filled_area(comparison, est_lat, est_lon, true_lat, true_lon, title)
            self.plot_violin_subplot(comparison, est_lat, est_lon, true_lat, true_lon, f'Violin Plots of Actual vs Estimated Locations ({title})')
            r2_lat = r2_score(comparison[true_lat], comparison[est_lat])
            r2_lon = r2_score(comparison[true_lon], comparison[est_lon])
            # MSE
            mse_lat= mean_squared_error(comparison[true_lat], comparison[est_lat])
            mse_lon = mean_squared_error(comparison[true_lon], comparison[est_lon])

            # RMSE (square root of MSE)
            rmse_lat = root_mean_squared_error(comparison[true_lat], comparison[est_lat])
            rmse_lon = root_mean_squared_error(comparison[true_lon], comparison[est_lon])
            print(f"MSE for Latitude ({title}): {mse_lat:.4f}")
            print(f"MSE for Longitude ({title}): {mse_lon:.4f}") 
            print(f"RMSE for Latitude ({title}): {rmse_lat:.4f}")
            print(f"RMSE for Longitude ({title}): {rmse_lon:.4f}") 
            print(f"R² for Latitude ({title}): {r2_lat:.4f}")
            print(f"R² for Longitude ({title}): {r2_lon:.4f}") 

    @staticmethod
    def analyze_comparison(df, estimated_lat_col, estimated_lon_col, actual_lat_col, actual_lon_col, title):
        """
        Analyze and visualize comparison between estimated and actual locations.

        Args:
            df (pd.DataFrame): DataFrame containing comparison data.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the analysis plots.

        Side effects:
            Creates and saves analysis plots.
            Modifies input DataFrame by adding and removing error columns.
        """
        df['lat_error'] = df[estimated_lat_col] - df[actual_lat_col]
        df['lon_error'] = df[estimated_lon_col] - df[actual_lon_col]
        df['MAE_lat'] = df['lat_error'].abs()
        df['MAE_lon'] = df['lon_error'].abs()
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df[actual_lat_col], df[actual_lon_col], label='Actual', color='blue', alpha=0.8)
        plt.scatter(df[estimated_lat_col], df[estimated_lon_col], label='Estimated', color='red', alpha=0.2)
        plt.title(f'Actual vs Estimated Locations ({title})')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['MAE_lat'], bins=30, color='skyblue', kde=True)
        plt.title(f'Distribution of Latitude Errors ({title})')
        plt.xlabel('Latitude Error')
       
        plt.tight_layout()
        plt.savefig(f'./Plots/Triangulation/{title}_analyze.png')
        plt.close()
        df.drop(columns=['lat_error','lon_error','MAE_lat','MAE_lon'], inplace=True)

    @staticmethod
    def plot_comparison(df, estimated_lat_col, estimated_lon_col, actual_lat_col, actual_lon_col, title, a=1, b=1):
        """
        Plot comparison between estimated and actual locations using spline interpolation.

        Args:
            df (pd.DataFrame): DataFrame containing comparison data.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the plot.
            a (float, optional): Alpha value for estimated data. Defaults to 1.
            b (float, optional): Alpha value for actual data. Defaults to 1.

        Side effects:
            Creates and saves a comparison plot.
        """
        x = np.arange(len(df))
        spline_lat = make_interp_spline(x, df[actual_lat_col])
        spline_lon = make_interp_spline(x, df[actual_lon_col])
        spline_est_lat = make_interp_spline(x, df[estimated_lat_col])
        spline_est_lon = make_interp_spline(x, df[estimated_lon_col])

        x_new = np.linspace(x.min(), x.max(), 300)
        
        plt.figure(figsize=(12, 5))
        plt.plot(x_new, spline_lat(x_new), color='blue', label='Actual Latitude', alpha=b)
        plt.plot(x_new, spline_lon(x_new), color='green', label='Actual Longitude',alpha=b)
        plt.plot(x_new, spline_est_lat(x_new), color='red', label='Estimated Latitude', alpha=a)
        plt.plot(x_new, spline_est_lon(x_new), color='orange', label='Estimated Longitude', alpha=a)

        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Coordinates')
        plt.legend()
        plt.grid()
        plt.savefig(f'./Plots/Triangulation/{title}_compariosn.png')
        plt.close()       

    @staticmethod
    def plot_filled_area(df, estimated_lat_col, estimated_lon_col, actual_lat_col, actual_lon_col, title):
        """
        Plot filled area between estimated and actual latitude values.

        Args:
            df (pd.DataFrame): DataFrame containing comparison data.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the plot.

        Side effects:
            Creates and saves a filled area plot.
        """
        x = np.arange(len(df))
        
        plt.figure(figsize=(12, 5))
        plt.fill_between(x, df[actual_lat_col], df[estimated_lat_col], color='gray', alpha=1, label='Area between Actual and Estimated')
        plt.plot(x, df[actual_lat_col], color='blue', label='Actual Latitude',alpha=0.1)
        plt.plot(x, df[estimated_lat_col], color='orange', label='Estimated Latitude',alpha=0.1)
        
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Latitude')
        plt.legend()
        plt.savefig(f'./Plots/Triangulation/{title}_filled_area.png')
        plt.close()        

    @staticmethod
    def plot_violin_subplot(df, estimated_lat_col, estimated_lon_col, actual_lat_col, actual_lon_col, title):
        """
        Create violin plots comparing estimated and actual latitude and longitude distributions.

        Args:
            df (pd.DataFrame): DataFrame containing comparison data.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the plot.

        Side effects:
            Creates and saves violin plots.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        sns.violinplot(data=[df[actual_lat_col], df[estimated_lat_col]], palette=['blue', 'red'], ax=axes[0])
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Actual Latitude', 'Estimated Latitude'])
        axes[0].set_title('Violin Plot of Latitude')
        axes[0].set_ylabel('Latitude')
        
        sns.violinplot(data=[df[actual_lon_col], df[estimated_lon_col]], palette=['blue', 'red'], ax=axes[1])
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Actual Longitude', 'Estimated Longitude'])
        axes[1].set_title('Violin Plot of Longitude')
        axes[1].set_ylabel('Longitude')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f'./Plots/Triangulation/{title}_violin.png')
        plt.close()        

def main():
    """
    Main function to run the triangulation estimation and analysis process.

    Side effects:
        Loads and processes data.
        Creates TriangulationEstimator and TriangulationResultAnalyzer objects.
        Runs estimations and analyses.
        Prints execution times.
    """
    start_time = time.time()

    data_path = './training-data/DEBI_Data_V2.xlsx'
    df = load_data(data_path)
    df = clean_data(df, drop=False)
    print(f"Data processing completed in {time.time() - start_time:.2f} seconds")

    # Estimate tower locations
    estimator_start_time = time.time()
    estimator = TriangulationEstimator(df)
    estimator.run_all_estimations()
    print(f"Estimation completed in {time.time() - estimator_start_time:.2f} seconds")

    # Analyze results
    analyzer_start_time = time.time()
    analyzer = TriangulationResultAnalyzer(df, estimator)
    analyzer.prepare_comparisons()
    analyzer.run_analysis()
    print(f"Analysis completed in {time.time() - analyzer_start_time:.2f} seconds")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
