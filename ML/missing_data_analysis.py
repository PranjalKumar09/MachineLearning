from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MissingValueAnalysisTemplate(ABC):
    @abstractmethod
    def analyze_missing_values(self, df: pd.DataFrame) -> None:
        """Abstract method to perform missing value analysis"""
        
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    
    def identify_missing_values(self, df: pd.DataFrame):
        """Identifies missing values in the dataframe"""
        pass
   
    def visualize_missing_values(self, df: pd.DataFrame):
        """Visualizes missing values using seaborn heatmap"""
        pass


class SimpleMissingValueAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """Identifies missing values in the dataframe using pandas isnull() method"""
        missing_values = df.isnull().sum()
        print("\n Missing values Count by Columns:")
        print(missing_values[missing_values > 0])
    
    def visualize_missing_values(self, df: pd.DataFrame):
        """Visualizes missing values using seaborn heatmap"""
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.show()
        
if __name__ == '__main__':
    pass


""" 
Missing values can be structured missing or randomly missing

"""