from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Perform bivariate analysis on two features of dataframe
        
        Parameters
        df (pd.DataFrame)
        feature1 (str)
        feature2 (str)
        
        Returns: None
        """
        pass


class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Plots the relationship  between two numerical features
        
        Parameters
        df (pd.DataFrame)
        feature1 (str)
        feature2 (str)
        
        Returns: None
        """
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1, y=feature2, data = df)
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str) -> None:
        """
        Plots the relationship between two categorical features
        
        Parameters
        df (pd.DataFrame)
        feature1 (str)
        feature2 (str)
        
        Returns: None
        """
        plt.figure(figsize=(10,6))
        sns.boxplot(x = feature1, y = feature2, data = df)
        plt.title(f"Relationship between {feature1} and {feature2}")
        plt.xlabel(feature1)
        plt.ylabel("Frequency")
        plt.show()
        
    

"""
In numerical vs numerical analysis , we can see outliers also we can see wheteher strong relationship or not (dense point)




"""