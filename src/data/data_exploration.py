import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


class DataExplorer:
    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, but got {type(data)}")

        self.data = data

    def plot_individual_variables(self) -> None:
        data = self.data

        # Distributions
        data.hist(bins=50, figsize=(20, 20))

        # Figure for Box plots
        fig, axes = plt.subplots(nrows=4, ncols=7)
        i = 0
        j = 0
        for column in data.columns:
            axes[i, j].boxplot(data[column])
            i = i + 1
            j = j + 1
            if i == 4:
                i = 0
            if j == 7:
                j = 0

    def plot_pollution(self) -> None:
        """Plots the target variables w.r.t. days based on the data in the object.
        """
        data = self.data

        # Create subplots: 1 row, 2 columns (1 for NO2, 1 for O3)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True)

        # Select just two years
        data = data[data.index < 731]

        # Create x-axis tick labels
        season_starts = [0, 80, 172, 264, 355, 445, 537, 629, 720, 810]
        season_names = ["Winter", "Spring", "Summer", "Fall", "Winter", "Spring", "Summer", "Fall", "Winter", "Spring"]

        # NO2 column
        axes[0].plot(data.index, data["Average_no2"], label="Average NO2")
        axes[0].set_xticks(season_starts)
        axes[0].set_xticklabels(season_names)
        axes[0].set_title("NO2 levels in Utrecht")
        axes[0].set_ylabel("NO2 level in μg/m3")
        axes[0].set_xlabel("Day of the year")
        axes[0].grid(True)

        # O3 column
        axes[1].plot(data.index, data["Average_o3"], label="Average O3")
        axes[0].set_xticks(season_starts)
        axes[0].set_xticklabels(season_names)
        axes[1].set_title("O3 levels in Utrecht")
        axes[1].set_ylabel("O3 level in μg/m3")
        axes[1].set_xlabel("Day of the year")
        axes[1].grid(True)

        # Adjust layout
        plt.tight_layout()

    def correlation_matrix(self) -> None:
        """ Creates a correlation heatmap based on the data.
        Used code from https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering
        """
        data = self.data

        # Rank correlation coefficients
        correlations = data.corr()
        ranked_no2 = abs(correlations['Average_no2']).sort_values(ascending=False)
        print("Ranked NO2:")
        print(ranked_no2)
        ranked_o3 = abs(correlations['Average_o3']).sort_values(ascending=False)
        print("Ranked O3:")
        print(ranked_o3)

        # Only keeping first ten for each ranking
        data = data[["FG", "VVN", "FHVEC", "FXX", "FHX", "UX", "FHN", "T10N", "UG", "TN", "UN", "EV24", "Q", "VVX",
                     "SQ", "TX", "TG"]]

        # Plot correlation matrix
        plt.figure(figsize=(15, 10))
        correlations = data.corr()
        sns.heatmap(round(correlations, 2), cmap='RdBu', annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1)

    def create_dendrogram(self) -> None:
        """Creates a dendrogram based on the given data
        Used code from https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering
        """
        data = self.data

        # Remove unnecessary column
        data = data.drop(columns=["YYYYMMDD"])

        # Create correlation matrix, make into a dissimilarity matrix, create dendogram
        correlations = data.corr()
        plt.figure(figsize=(12, 6))
        dissimilarity = 1 - abs(correlations)
        z = linkage(squareform(dissimilarity), 'complete')
        dendrogram(z, labels=data.columns, orientation='top', leaf_rotation=90)

    def inspect_threshold_hierarchical_clustering(self) -> None:
        """ Creates multiple plots showing the influence of the threshold function,
            can be used to determine the threshold value in correlation_heatmap_hierarchical_clustering()
        Used code from https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering
        """
        data = self.data

        # Remove unnecessary column
        data = data.drop(columns=["YYYYMMDD"])

        plt.figure(figsize=(15, 10))

        correlations = data.corr()
        dissimilarity = 1 - abs(correlations)
        z = linkage(squareform(dissimilarity), 'complete')

        for index, t in enumerate(np.arange(0.2, 1.1, 0.1)):

            # Subplot idx + 1
            plt.subplot(3, 3, index+1)

            # Calculate the cluster
            labels = fcluster(z, t, criterion='distance')

            # Keep the indices to sort labels
            labels_order = np.argsort(labels)

            # Build a new dataframe with the sorted columns
            for idx, i in enumerate(data.columns[labels_order]):
                if idx == 0:
                    clustered = pd.DataFrame(data[i])
                else:
                    df_to_append = pd.DataFrame(data[i])
                    clustered = pd.concat([clustered, df_to_append], axis=1)

            # Plot the correlation heatmap
            correlations = clustered.corr()
            sns.heatmap(round(correlations, 2), cmap='RdBu', vmin=-1, vmax=1,
                        xticklabels=False, yticklabels=False)
            plt.title("Threshold = {}".format(round(t, 2)))

    def correlation_heatmap_hierarchical_clustering(self, threshold: float) -> None:
        """Clusters a correlation heatmap and creates a figure of this.
        Used code from https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering

        Args:
            threshold (float): The threhold used to cluster the heatmap
        """
        data = self.data

        # Remove unnecessary column
        data = data.drop(columns=["YYYYMMDD"])

        correlations = data.corr()
        dissimilarity = 1 - abs(correlations)
        z = linkage(squareform(dissimilarity), 'complete')

        # Clusterize the data
        labels = fcluster(z, threshold, criterion='distance')

        labels_order = np.argsort(labels)

        # Build a new dataframe with the sorted columns
        for idx, i in enumerate(data.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(data[i])
            else:
                df_to_append = pd.DataFrame(data[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)

        correlations = clustered.corr()
        # Create heatmap with dendrogram
        sns.clustermap(correlations, figsize=(20, 15), method="complete", cmap='RdBu', annot=True,
                       annot_kws={"size": 7}, vmin=-1, vmax=1)
        # Create heatmap without dendrogram
        plt.figure(figsize=(20, 15))
        sns.heatmap(round(correlations, 2), cmap='RdBu', annot=True,
                    annot_kws={"size": 7}, vmin=-1, vmax=1)

    def _plot_variance(self, pca: PCA) -> None:
        # Create figure of pca
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(
            xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
        )
        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(
            xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
        )
        # Set up figure
        fig.set(figwidth=8, dpi=100)
        return axs

    def pca_analysis(self) -> pd.Series:
        """Analyses the data and returns the components, allong with their explained variance. Also plots this data.

        Returns:
            pd.Series: The explained variance w.r.t. each component cumulatively summed.
        """
        data = self.data

        # Scale dataset and remove truth values
        X = data.copy()
        y_no2 = X.pop("Average_no2")
        y_o3 = X.pop("Average_o3")

        scaler = MinMaxScaler()
        columns = X.columns
        X[columns] = scaler.fit_transform(data[columns])

        # Create principal components
        pca = PCA()
        X_pca = pca.fit_transform(X)

        # Convert to dataframe
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)

        # This does not work and I don't know why, but it is useful later on probably so I'm keeping it
        # loadings = pd.DataFrame(
        #     pca.components_.T,  # transpose the matrix of loadings
        #     columns=component_names,  # so the columns are the principal components
        #     index=data.columns,  # and the rows are the original features
        # )

        self._plot_variance(pca)
        results = np.cumsum(pca.explained_variance_ratio_)

        # This may be useful still so I'm keeping it
        # results = pd.concat({
        #     "NO2_scores": self._make_mi_scores(X_pca, y_no2, discrete_features=False),
        #     "O3_scores": self._make_mi_scores(X_pca, y_o3, discrete_features=False)
        # }, axis=1, join="inner")
        # results["Rank"] = ((results["NO2_scores"] + results["O3_scores"])
        #                 .astype(float).rank(method='dense', ascending=False).astype(int))
        # results = results.sort_values('Rank')
        # results["Cumulative_score"] = (results["NO2_scores"] + results["O3_scores"]).cumsum()
        # results["Cumulative percentage"] = results["Cumulative_score"] / results["Cumulative_score"].iloc[-1]
        return results

    def show_missing_values(self) -> None:
        """_summary_

        Returns:
            int: _description_
        """
        missing_values = self.data.isnull().sum()
        print("Missing data values:")
        print(missing_values)

    def analyse_individual_variables(self) -> None:
        """_summary_
        """
        print("Data description:")
        print(self.data.describe())
        self.plot_individual_variables()
        self.plot_pollution()

    def create_plots(self) -> None:
        """Function to plot all previous functions.
        """
        plt.tight_layout()
        plt.show()


def pca_component_ranking_using_random_forest_plot(x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    feature_names = [f"component_{i+1}" for i in range(41)]
    rf = RandomForestRegressor(random_state=0)
    rf.fit(x_train, y_train)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    plt.show()
