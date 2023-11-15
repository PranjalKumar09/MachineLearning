from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MultivariateAnalysisStrategy(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform comprehensive multivariate analysis by generating correlation heatmap and pair plot
        
        Parameters:
        df (pd.DataFrame)
        
        Return: None
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        
    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display heatmap of the correlations between
        
        Parameters:
        df (pd.DataFrame)
        
        Returns: None
        """
        pass
    
    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display pairwise relationships between variables in the dataset
        
        Parameters:
        df (pd.DataFrame)
        
        Returns: None
        """
        pass
class SimpleMultivariateAnalysis(MultivariateAnalysisStrategy):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display heatmap of the correlations between
        
        Parameters:
        df (pd.DataFrame)
        
        Returns: None
        """
        plt.figure(figsize=(12,10))
        sns.heatmap( df.corr(),  annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
    
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and display pair plotg for selected feature    
    
        Parameters:
        df (pd.DataFrame)
        
        Returns: None
        """
        sns.pairplot(df)
        plt.show()
        
        

