# ingest data

import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path, str) -> pd.DataFrame:
        """Abstract methods to ingest data from given file  """
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str, str: str) -> pd.DataFrame:
        """Extract a .zip file & return content as pandas DataFrame"""
        
        if not file_path.endswith('.zip'):
            raise ValueError('Input file must be a .zip file')
        
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
            
        extracted_files = os.listdir("extracted_data")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]
        
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in extracted_data")
        
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in extracted_data")
    
        csv_file_path = os.path.join("extracted_data", csv_files)
        df = pd.read_csv(csv_file_path)
        
        return df
    
class DataIngestorFactory:
    @staticmethod
    def create_data_ingestor(file_path: str) -> DataIngestor:
        """Factory method to create an appropriate DataIngestor instance based on file extension"""
        
        if file_path.endswith('.zip'):
            return ZipDataIngestor()
        else:
            raise ValueError('Unsupported file extension')
        
    
if __name__ == '__main__':
    pass    