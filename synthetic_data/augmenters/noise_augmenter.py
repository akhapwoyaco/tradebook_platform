import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from loguru import logger

class NoiseAugmenter:
    """
    A class for applying various types of noise augmentation to time-series data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('augmentation', {})
        self.noise_type = self.config.get('noise_type', 'gaussian')
        # For Gaussian/Uniform: magnitude as a percentage (e.g., 0.01 for 1%) of std/range.
        # For Salt-and-Pepper: magnitude as a ratio (e.g., 0.01 for 1% of points).
        self.noise_magnitude = self.config.get('noise_magnitude', 0.01) 
        self.target_columns = self.config.get('target_columns', [])
        logger.info(f"NoiseAugmenter initialized with noise_type='{self.noise_type}', magnitude={self.noise_magnitude}")

    def apply_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies noise augmentation to specified numerical columns in a DataFrame.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            
        Returns:
            pd.DataFrame: The DataFrame with noise applied.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty, returning as is.")
            return df.copy()

        augmented_df = df.copy()
        
        # If no target columns specified, apply to all numeric columns
        if not self.target_columns:
            target_cols = augmented_df.select_dtypes(include=np.number).columns.tolist()
            logger.info(f"No specific target columns provided. Applying noise to all numeric columns: {target_cols}")
        else:
            target_cols = [col for col in self.target_columns if col in augmented_df.columns and pd.api.types.is_numeric_dtype(augmented_df[col])]
            if not target_cols:
                logger.warning(f"No valid numeric target columns found for augmentation in {self.target_columns}. Returning original DataFrame.")
                return df.copy()
            logger.info(f"Applying noise to specified numeric columns: {target_cols}")

        for col in target_cols:
            if self.noise_type == 'gaussian':
                # Magnitude as a percentage of the column's standard deviation
                std_dev = augmented_df[col].std()
                if std_dev == 0 or pd.isna(std_dev):
                    logger.warning(f"Standard deviation for column '{col}' is zero or NaN, skipping Gaussian noise.")
                    continue
                noise_std = std_dev * self.noise_magnitude
                noise = np.random.normal(0, noise_std, size=len(augmented_df))
                augmented_df[col] += noise
                logger.debug(f"Applied Gaussian noise to '{col}' with std {noise_std:.4f}")
            elif self.noise_type == 'uniform':
                # Magnitude as a percentage of the column's range
                col_min = augmented_df[col].min()
                col_max = augmented_df[col].max()
                col_range = col_max - col_min
                if col_range == 0 or pd.isna(col_range):
                    logger.warning(f"Range for column '{col}' is zero or NaN, skipping Uniform noise.")
                    continue
                noise_range = col_range * self.noise_magnitude
                noise = np.random.uniform(-noise_range / 2, noise_range / 2, size=len(augmented_df))
                augmented_df[col] += noise
                logger.debug(f"Applied Uniform noise to '{col}' with range {noise_range:.4f}")
            elif self.noise_type == 'salt_and_pepper':
                # Applies extreme values to a small percentage of data points
                # noise_magnitude now represents the ratio of points to alter
                ratio = self.noise_magnitude 
                num_corrupt = int(np.ceil(ratio * len(augmented_df)))
                if num_corrupt == 0: 
                    logger.debug(f"No points to corrupt for Salt-and-Pepper noise on '{col}' (ratio {ratio}).")
                    continue

                idx_to_corrupt = np.random.choice(augmented_df.index, num_corrupt, replace=False)
                
                min_val = augmented_df[col].min()
                max_val = augmented_df[col].max()
                
                # Introduce 50% salt (max-like), 50% pepper (min-like)
                salt_idx = np.random.choice(idx_to_corrupt, int(num_corrupt * 0.5), replace=False)
                pepper_idx = np.setdiff1d(idx_to_corrupt, salt_idx)

                # Ensure a meaningful offset, even if min/max are zero
                # Use a small fraction of the column's range for the "perturbation" around min/max
                value_range = max_val - min_val
                if value_range == 0:
                    # If all values are the same, use a small absolute value for perturbation
                    # This handles cases where column is all zeros, or all ones, etc.
                    perturbation_amount = 0.1 * np.mean(np.abs(augmented_df[col].values)) # Use mean of abs values as a scale
                    if perturbation_amount == 0: # Fallback for all zeros
                        perturbation_amount = 0.1 
                else:
                    perturbation_amount = value_range * 0.1 # 10% of the range for perturbation

                # Apply salt (values near max_val + perturbation)
                augmented_df.loc[salt_idx, col] = max_val + perturbation_amount * np.random.rand(len(salt_idx))
                # Apply pepper (values near min_val - perturbation)
                augmented_df.loc[pepper_idx, col] = min_val - perturbation_amount * np.random.rand(len(pepper_idx))
                
                logger.debug(f"Applied Salt-and-Pepper noise to '{col}' on {num_corrupt} points.")
            else:
                logger.warning(f"Unknown noise type: {self.noise_type}. Skipping augmentation for column '{col}'.")

        logger.info(f"Augmentation completed for {len(augmented_df)} samples using '{self.noise_type}' noise.")
        return augmented_df

# Example Usage
if __name__ == "__main__":
    # Create a dummy DataFrame
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'price': np.sin(np.linspace(0, 10, 100)) * 10 + 50,
        'amount': np.random.randint(100, 500, 100),
        'type': ['a'] * 50 + ['b'] * 50
    }
    df = pd.DataFrame(data).set_index('date')
    print("Original DataFrame head:\n", df.head())
    print("Original DataFrame description:\n", df.describe())

    # Example 1: Gaussian noise on specific columns
    config_gaussian = {
        'augmentation': {
            'noise_type': 'gaussian',
            'noise_magnitude': 0.05, # 5% of std dev
            'target_columns': ['price', 'amount']
        }
    }
    augmenter_gaussian = NoiseAugmenter(config_gaussian)
    augmented_df_gaussian = augmenter_gaussian.apply_augmentation(df.copy())
    print("\nAugmented with Gaussian Noise head:\n", augmented_df_gaussian.head())
    print("Augmented with Gaussian Noise description:\n", augmented_df_gaussian.describe())

    # Example 2: Uniform noise on all numeric columns
    config_uniform = {
        'augmentation': {
            'noise_type': 'uniform',
            'noise_magnitude': 0.02, # 2% of range
            'target_columns': [] # Apply to all numeric
        }
    }
    augmenter_uniform = NoiseAugmenter(config_uniform)
    augmented_df_uniform = augmenter_uniform.apply_augmentation(df.copy())
    print("\nAugmented with Uniform Noise head:\n", augmented_df_uniform.head())
    print("Augmented with Uniform Noise description:\n", augmented_df_uniform.describe())

    # Example 3: Salt-and-pepper noise on a single column
    config_salt_pepper = {
        'augmentation': {
            'noise_type': 'salt_and_pepper',
            'noise_magnitude': 0.02, # 2% of points corrupted
            'target_columns': ['price']
        }
    }
    augmenter_salt_pepper = NoiseAugmenter(config_salt_pepper)
    augmented_df_sp = augmenter_salt_pepper.apply_augmentation(df.copy())
    print("\nAugmented with Salt-and-Pepper Noise head:\n", augmented_df_sp.head())
    print("Augmented with Salt-and-Pepper Noise description:\n", augmented_df_sp.describe())

    # Check a corrupted point
    # Find an index that was likely corrupted (random, so may not be index 10 every time)
    # This is for illustrative purposes; in real tests, you'd check based on actual changes.
    original_price_at_10 = df.loc[df.index[10], 'price']
    sp_price_at_10 = augmented_df_sp.loc[augmented_df_sp.index[10], 'price']
    print(f"\nOriginal price at index 10: {original_price_at_10:.2f}")
    print(f"Salt-and-Pepper price at index 10: {sp_price_at_10:.2f}")

    # Example 4: Test Salt-and-Pepper with a column of all zeros
    data_zeros = {
        'date': pd.date_range(start='2023-01-01', periods=10, freq='H'),
        'value': np.zeros(10)
    }
    df_zeros = pd.DataFrame(data_zeros).set_index('date')
    print("\nOriginal DataFrame (all zeros) head:\n", df_zeros.head())

    config_sp_zeros = {
        'augmentation': {
            'noise_type': 'salt_and_pepper',
            'noise_magnitude': 0.3, # 30% of points corrupted
            'target_columns': ['value']
        }
    }
    augmenter_sp_zeros = NoiseAugmenter(config_sp_zeros)
    augmented_df_sp_zeros = augmenter_sp_zeros.apply_augmentation(df_zeros.copy())
    print("Augmented with Salt-and-Pepper Noise (all zeros) head:\n", augmented_df_sp_zeros.head())