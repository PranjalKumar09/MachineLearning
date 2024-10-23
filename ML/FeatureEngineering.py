from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import RobustScaler


# Abstract Base Class
class FeatureEngineeringStrategy(ABC):
    
    @abstractmethod
    def apply_transformation(self, data):
        """
        Abstract method that must be implemented by all subclasses.
        It takes the data as input and returns the transformed data.
        """
        pass

class LogTransformation(FeatureEngineeringStrategy):
    
    def apply_transformation(self, data):
        """
        Applies log transformation to the data.
        It assumes that the data is a pandas DataFrame or Series and handles non-positive values.
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return np.log1p(data.clip(lower=0))  # log(1 + data), clipping to avoid negatives.
        else:
            raise TypeError("Data should be a pandas DataFrame or Series.")


class StandardScaling(FeatureEngineeringStrategy):
    
    def __init__(self):
        self.scaler = StandardScaler()

    def apply_transformation(self, data):
        """
        Applies standard scaling to the data.
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
        else:
            raise TypeError("Data should be a pandas DataFrame.")

class OneHotEncoding(FeatureEngineeringStrategy):
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid multicollinearity

    def apply_transformation(self, data):
        """
        Applies one-hot encoding to categorical features.
        """
        if isinstance(data, pd.DataFrame):
            encoded_data = pd.DataFrame(self.encoder.fit_transform(data),
                                        columns=self.encoder.get_feature_names_out(data.columns))
            return encoded_data
        else:
            raise TypeError("Data should be a pandas DataFrame.")


class MinMaxScaling(FeatureEngineeringStrategy):
    
    def __init__(self, feature_range=(0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, data):
        """
        Applies Min-Max scaling to the data.
        """
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
        else:
            raise TypeError("Data should be a pandas DataFrame.")



class PowerTransformation(FeatureEngineeringStrategy):
    
    def __init__(self, method='yeo-johnson'):  # 'box-cox' also available, but it only works for positive data
        self.transformer = PowerTransformer(method=method)

    def apply_transformation(self, data):
        """
        Applies a power transformation to the data.
        """
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(self.transformer.fit_transform(data), columns=data.columns)
        else:
            raise TypeError("Data should be a pandas DataFrame.")


class Binarization(FeatureEngineeringStrategy):
    
    def __init__(self, threshold=0.0):
        self.binarizer = Binarizer(threshold=threshold)

    def apply_transformation(self, data):
        """
        Applies binarization to the data.
        """
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(self.binarizer.fit_transform(data), columns=data.columns)
        else:
            raise TypeError("Data should be a pandas DataFrame.")


class RobustScaling(FeatureEngineeringStrategy):
    
    def __init__(self):
        self.scaler = RobustScaler()

    def apply_transformation(self, data):
        """
        Applies robust scaling to the data (less sensiti\ve to outliers).
        """
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
        else:
            raise TypeError("Data should be a pandas DataFrame.")
        
        
        
"""
1. LogTransformation

    What it does: Applies a log transformation to compress the range of data.
    When to use: Use when data is heavily skewed or when you want to reduce the impact of large values (e.g., for non-negative features like prices or counts).
    ML Use: Helps stabilize variance and make data more normally distributed, beneficial in linear models and regressions.

2. StandardScaling

    What it does: Scales data to have a mean of 0 and a standard deviation of 1 (Z-score normalization).
    When to use: Use when features have different units or distributions (e.g., in linear models, SVM, or KNN).
    ML Use: Ensures that features are on a similar scale, which is crucial for gradient-based algorithms (like logistic regression or neural networks).

3. OneHotEncoding

    What it does: Converts categorical variables into binary (0/1) columns for each category.
    When to use: Use when you have categorical features and algorithms that require numerical input (e.g., decision trees, random forests, or linear models).
    ML Use: Prevents algorithms from assuming an ordinal relationship between categories.

4. MinMaxScaling

    What it does: Scales features to a fixed range, typically [0, 1].
    When to use: Use when you want to maintain the relative distribution of features (e.g., in algorithms like KNN, neural networks, or clustering).
    ML Use: Useful in models sensitive to the scale of data, like neural networks and distance-based algorithms.

5. PolynomialFeaturesTransformation

    What it does: Generates new features by raising existing features to a power (e.g., degree-2 polynomial).
    When to use: Use when you suspect non-linear relationships between features and target variables (e.g., linear regression, SVM).
    ML Use: Enhances linear models by allowing them to model more complex patterns.

6. PowerTransformation

    What it does: Stabilizes variance and makes data more normally distributed using transformations like Box-Cox or Yeo-Johnson.
    When to use: Use for features with non-normal distributions and to reduce skewness (e.g., in linear models and regression analysis).
    ML Use: Improves the performance of algorithms that assume normality in features, like linear regression and ANOVA.

7. Binarization

    What it does: Converts numerical features into binary values (0 or 1) based on a threshold.
    When to use: Use when you want to treat numeric features as binary (e.g., features representing yes/no or on/off states).
    ML Use: Simplifies features when you need a binary representation for classification tasks.

8. RobustScaling

    What it does: Scales features using the median and interquartile range, reducing sensitivity to outliers.
    When to use: Use when data contains outliers that could distort the scaling (e.g., in models like linear regression, KNN, or SVM).
    ML Use: Beneficial in situations with outliers, ensuring that they don't disproportionately affect scaling.

When to Use These Transformations in Machine Learning Models:

    LogTransformation: Use for positively skewed data (e.g., skewed continuous variables). Works well for regression models and linear models.
    StandardScaling: Essential for algorithms that assume normally distributed data (e.g., linear regression, logistic regression, SVM, and neural networks).
    OneHotEncoding: Always apply to categorical variables in models that can't handle categories directly (e.g., logistic regression, KNN, SVM, neural networks).
    MinMaxScaling: Use in models that require data in a specific range, especially for distance-based models like KNN, neural networks, or clustering algorithms.
    PolynomialFeaturesTransformation: Useful in models where relationships are nonlinear (e.g., for boosting linear regression models or improving SVMs).
    PowerTransformation: Apply to features with high variance and skewness; useful in linear regression and Gaussian-based models.
    Binarization: Use when you need a clear distinction between two states (e.g., binary classification tasks).
    RobustScaling: Use in the presence of outliers, especially when scaling for linear models, SVM, or tree-based methods where outliers are less impactful.
    
    
"""