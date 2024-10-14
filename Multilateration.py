import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from shapely.geometry import Point
import multiprocessing
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import time
from tqdm import tqdm
from data_utils import load_data, clean_data
tqdm.pandas()

class MultilaterationEstimator:
    """
    A class for estimating cell tower locations using multilateration techniques.
    
    This class implements two methods of multilateration:
    1. Geographical multilateration (geo)
    2. Site-based multilateration (site)
    
    Attributes:
        df (pandas.DataFrame): The input dataframe containing client and cell tower information.
        Multil_geo (pandas.DataFrame): Results of geographical multilateration.
        Multil_site (pandas.DataFrame): Results of site-based multilateration.
    """

    def __init__(self, df):
        """
        Initialize the MultilaterationEstimator with a dataframe.

        Args:
            df (pandas.DataFrame): The input dataframe containing client and cell tower information.
        """
        self.df = df
        self.Multil_geo = None
        self.Multil_site = None

    def run_all_estimations(self):
        """
        Run both geographical and site-based multilateration estimations.
        """
        print("Running multilateration (geo)...")
        self.run_multilateration_geo()
        print("Running multilateration (site)...")
        self.run_multilateration_site()

    def run_multilateration_geo(self):
        """
        Run geographical multilateration to estimate tower locations.
        """
        self.df['geometry'] = self.df.apply(lambda row: Point(row['client_longitude'], row['client_latitude']), axis=1)
        clients_by_tower = self.df.groupby("start_cell_id_a").apply(lambda group: group[["geometry", "dbm_a"]].values.tolist())
        
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(self.process_tower_location, clients_by_tower.items()), total=len(clients_by_tower)))
        
        self.Multil_geo = pd.DataFrame(results, columns=['cell_tower_id', 'Estimated_Tower_Lon', 'Estimated_Tower_Lat'])
        self.Multil_geo.set_index('cell_tower_id', inplace=True)

    @staticmethod
    def process_tower_location(item):
        """
        Process a single tower's location estimation.

        Args:
            item (tuple): A tuple containing tower_id and client information.

        Returns:
            list: A list containing the tower_id and estimated location.
        """
        tower_id, clients = item
        tower_location = MultilaterationEstimator.optimize_tower_location(clients)
        return [tower_id] + list(tower_location)

    def run_multilateration_site(self):
        """
        Run site-based multilateration to estimate tower locations.
        """
        self.df['estimated_distance'] = self.df['rsrp_a'].apply(self.estimate_distance_from_signal)
        
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(self.process_site_location, self.df.groupby('Site_ID')), total=len(self.df['Site_ID'].unique())))
        
        self.Multil_site = pd.DataFrame(results)

    @staticmethod
    def process_site_location(group):
        """
        Process a single site's location estimation.

        Args:
            group (tuple): A tuple containing site_id and group data.

        Returns:
            dict: A dictionary containing the site_id and estimated location.
        """
        site_id, group_data = group
        client_positions = group_data[['client_latitude', 'client_longitude']].values
        distances = group_data['estimated_distance'].values
        initial_guess = [group_data['client_latitude'].mean(), group_data['client_longitude'].mean()]
        result = least_squares(MultilaterationEstimator.residuals, initial_guess, args=(client_positions, distances))
        tower_position = result.x
        return {'Site_ID': site_id, 'Estimated_Tower_Lat': tower_position[0], 'Estimated_Tower_Lon': tower_position[1]}

    @staticmethod
    def optimize_tower_location(clients):
        """
        Optimize the tower location based on client information.

        Args:
            clients (list): A list of client information including geometry and signal strength.

        Returns:
            numpy.ndarray: The optimized tower location.
        """
        def multilateration(tower_location, clients):
            tower_point = Point(tower_location)
            squared_errors = []
            for client in clients:
                client_point = client[0]
                distance = client_point.distance(tower_point)
                estimated_distance = 1 / (np.abs(client[1]) + 1e-6)
                squared_errors.append((distance - estimated_distance) ** 2)
            return np.sum(squared_errors)

        initial_guess = np.array([
            np.mean([client[0].x for client in clients]),
            np.mean([client[0].y for client in clients])
        ])
        result = minimize(multilateration, initial_guess, args=(clients,))
        return result.x

    @staticmethod
    def estimate_distance_from_signal(rsrp, reference_rsrp=-50, path_loss_exponent=2):
        """
        Estimate the distance from a cell tower based on the RSRP signal strength.

        Args:
            rsrp (float): The Received Signal Reference Power.
            reference_rsrp (float): The reference RSRP at a known distance (default is -50).
            path_loss_exponent (float): The path loss exponent (default is 2).

        Returns:
            float: The estimated distance from the cell tower.
        """
        return 10 ** ((reference_rsrp - rsrp) / (10 * path_loss_exponent))

    @staticmethod
    def residuals(tower_position, client_positions, distances):
        """
        Calculate the residuals for least squares optimization.

        Args:
            tower_position (list): The estimated tower position [latitude, longitude].
            client_positions (numpy.ndarray): Array of client positions.
            distances (numpy.ndarray): Array of distances from clients to the tower.

        Returns:
            list: The residuals between estimated and actual distances.
        """
        lat, lon = tower_position
        residuals = []
        for i, client_pos in enumerate(client_positions):
            client_lat, client_lon = client_pos
            d = MultilaterationEstimator.haversine_distance(lat, lon, client_lat, client_lon)
            residual = d - distances[i]
            residuals.append(residual)
        return residuals

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points on Earth.

        Args:
            lat1, lon1 (float): Latitude and longitude of the first point.
            lat2, lon2 (float): Latitude and longitude of the second point.

        Returns:
            float: The distance between the two points in kilometers.
        """
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

class MultilaterationResultAnalyzer:
    """
    A class for analyzing and visualizing the results of multilateration estimation.

    This class provides methods to compare estimated tower locations with actual locations,
    generate various plots, and calculate performance metrics.

    Attributes:
        df (pandas.DataFrame): The original dataframe with client and tower information.
        estimator (MultilaterationEstimator): An instance of the MultilaterationEstimator class.
        comparisons (list): A list of comparison dataframes and associated metadata.
    """

    def __init__(self, df, estimator):
        """
        Initialize the MultilaterationResultAnalyzer.

        Args:
            df (pandas.DataFrame): The original dataframe with client and tower information.
            estimator (MultilaterationEstimator): An instance of the MultilaterationEstimator class.
        """
        self.df = df
        self.estimator = estimator
        self.comparisons = None

    def prepare_comparisons(self):
        """
        Prepare comparison dataframes for both geographical and site-based multilateration results.
        """
        long_lat_grouped_by_site_id = self.df.groupby('Site_ID')[['Latitude', 'Longitude']].mean().reset_index()
        long_lat_grouped_by_cell_id = self.df.groupby('start_cell_id_a')[['Latitude', 'Longitude']].mean().reset_index()

        comparison_Multi = pd.merge(self.estimator.Multil_site, long_lat_grouped_by_site_id, on='Site_ID', suffixes=('_est', '_true'))
        comparison_MultiGeo = pd.merge(self.estimator.Multil_geo, long_lat_grouped_by_cell_id, left_on='cell_tower_id', right_on='start_cell_id_a', suffixes=('_est', '_true'))
        
        self.comparisons = [
            (comparison_MultiGeo, 'Comparison MultiGeo', 'Estimated_Tower_Lat', 'Estimated_Tower_Lon', 'Latitude', 'Longitude'),
            (comparison_Multi, 'Comparison Multi', 'Estimated_Tower_Lat', 'Estimated_Tower_Lon', 'Latitude', 'Longitude'),
        ]
        comparison_MultiGeo.to_excel('./predictions/comparison_MultiGeo.xlsx', index=False)
        comparison_Multi.to_excel('./predictions/comparison_Multi.xlsx', index=False)

    def run_analysis(self):
        """
        Run the complete analysis, including various plots and R² score calculations.
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
        Analyze and plot the comparison between estimated and actual locations.

        Args:
            df (pandas.DataFrame): The comparison dataframe.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the plot.
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
        plt.savefig(f'./Plots/Multilateration/{title}_analyze.png')
        plt.close()
        df.drop(columns=['lat_error','lon_error','MAE_lat','MAE_lon'], inplace=True)

    @staticmethod
    def plot_comparison(df, estimated_lat_col, estimated_lon_col, actual_lat_col, actual_lon_col, title, a=1, b=1):
        """
        Plot a comparison of estimated and actual locations.

        Args:
            df (pandas.DataFrame): The comparison dataframe.
            estimated_lat_col (str): Column name for estimated latitude.
            estimated_lon_col (str): Column name for estimated longitude.
            actual_lat_col (str): Column name for actual latitude.
            actual_lon_col (str): Column name for actual longitude.
            title (str): Title for the plot.
            a (float): Alpha value for estimated location lines.
            b (float): Alpha value for actual location lines.
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
        plt.savefig(f'./Plots/Multilateration/{title}_compariosn.png')
        plt.close()                
        #plt.show()

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
        plt.savefig(f'./Plots/Multilateration/{title}_filled_area.png')
        plt.close()                
        #plt.show()

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
        plt.savefig(f'./Plots/Multilateration/{title}_violin.png')
        plt.close()                
        #plt.show()


def main():
    """
    Main function to run the Multilateration estimation and analysis process.

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
    estimator = MultilaterationEstimator(df)
    estimator.run_all_estimations()
    print(f"Estimation completed in {time.time() - estimator_start_time:.2f} seconds")

    # Analyze results
    analyzer_start_time = time.time()
    analyzer = MultilaterationResultAnalyzer(df, estimator)
    analyzer.prepare_comparisons()
    analyzer.run_analysis()
    print(f"Analysis completed in {time.time() - analyzer_start_time:.2f} seconds")

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
