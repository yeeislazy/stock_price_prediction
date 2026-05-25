from utils.logger import get_logger
import pandas as pd

def split_train_test(df: pd.DataFrame, test_mode=None, test_size=0.2):
    logger = get_logger(__name__)
    df = df.dropna().reset_index(drop=True)
    if test_mode == 'ratio':
        if not isinstance(test_size, float):
            logger.warning("Invalid test_size for ratio mode. It should be a float between 0 and 1. Defaulting to 80% train and 20% test split.")
            test_size = 0.2
        if test_size > 1 or test_size < 0:
            logger.warning("Invalid test_size for ratio mode. It should be a float between 0 and 1. Defaulting to 80% train and 20% test split.")
            test_size = 0.2
        split_idx = int(len(df) - len(df) * test_size)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
    elif test_mode == 'period':
        if not isinstance(test_size, int):
            logger.warning("Invalid test_size for period mode. It should be an integer representing the number of periods. Defaulting to 80% train and 20% test split.")
            test_size = 14
        train_df = df.iloc[:-(test_size+60)].reset_index(drop=True)
        test_df = df.iloc[-(test_size+60):].reset_index(drop=True)
    else:
        train_df = df
        test_df = None
    
    train_df = train_df.dropna().reset_index(drop=True)
    if test_df is not None:
        test_df = test_df.dropna().reset_index(drop=True)
    
    return train_df, test_df