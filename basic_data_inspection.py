from abc import ABC, abstractmethod

import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame) -> None:
        """Inspect the data and print relevant information
        
        Parameters
            df (pd.DataFrame): Dataframe on which inspection
            
        Returns
            None: This method prints the inspection results directly
        """
        pass
    
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """ Inspects and prints the data types and non=null counts
        
        Parameters:
            df (pd.DataFrame)
        """
        print("Data Types:")
        print(df.dtypes)
        print("\nNon-null Counts:")
        print(df.count())
        print("\nData types and non null Counts:")
        print(df.info())        

class SummaryStatisticsInspectionStaregy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Print summary statistics for numerical and categorical 
        
        Parameters:
            df (pd.DataFrame)
        Returns:
            Prints summary statistics
        """
        print("Summary Statistics (Numerical Features:")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features:")
        print(df.describe(include=["O"]))
        
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """ 
        Initiliazes DataInspector with specific inspection
        
        Parameters:
            strategy (DataInspectionStrategy)
        
        Returns: None
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataInspectionStrategy):
        """ 
        Sets the new strategy for the DataInspector
        
        Parameters:
            strategy (DataInspectionStrategy)
        
        Returns: None
        """
        self._strategy = strategy
        
    def execute_inspection(self, df: pd.DataFrame):
        """ 
        Executes the current strategy on the provided dataframe
        
        Parameters:
            df (pd.DataFrame)
        
        Returns: None
        """
        self._strategy.inspect(df)
        
if __name__ == "__main__":
    pass