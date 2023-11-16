from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str) -> None:
        """Perform univariate analysis on specific feature of dataframe
        
        Parameters
        df (pd.DataFrame)
        feature (str)
        
        Returns: None
        """
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of numerical feautre using histograms and KDE
        
        Parameters:
        df (pd.DataFrame)
        feature (str)

        Returns: None
        """
        
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel("Feature")
        plt.ylabel("Frequency")
        plt.show()
    
class CateoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the count of categorical feature using countplot
        
        Parameters:
        df (pd.DataFrame)
        feature (str)

        Returns: None
        """
        
        plt.figure(figsize=(10,6))
        sns.countplot(x = df[feature], data = df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel("Feature")
        plt.ylabel("Frequency")
        plt.show()
        

if __name__ == "__main__":
    pass
        
"""

In numerical analysis , if majority data lies on certain range we like between x1 and x2 we can think of applying log transformation to normalizae distribution


In categorical analysis , if we have more common values in less categories that that means these data have significant effect
"""


