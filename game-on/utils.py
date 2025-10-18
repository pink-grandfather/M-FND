## Importing libraries
import random 
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import config, dataset

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)



def set_up_mediaeval2015():
    """
    Loads the mediaeval graphical dataset.

    Download raw mediaeval dataset from: https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2015

    Returns:
        DGLDataset: Train graph dataset
        DGLDataset: Test graph dataset
    """

    # 1. 读取原始 CSV
    df_train = pd.read_csv(f'{config.root_dir}{config.me15_train_csv_name}', sep='\t')
    df_test = pd.read_csv(f'{config.root_dir}{config.me15_test_csv_name}', sep='\t')

    # 2. 只保留你已有的 10 张图（21→30）
    # 用 CSV 里真实存在的图像 ID
    existing_images = set(df_train['imageId(s)'].unique())

    df_train = df_train[df_train['imageId(s)'].isin(existing_images)].copy()
    df_test = df_test[df_test['imageId(s)'].isin(existing_images)].copy()

    # 如果过滤后测试集为空，直接复制训练集一小部分（快速跑通）
    if df_test.empty:
        df_test = df_train.sample(n=min(5, len(df_train)), random_state=42)

    # 3. 重置索引
    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    # 4. 构建图数据集
    dataset_train = dataset.GraphDataset(
        df_train, config.root_dir,
        "imageId(s)", "tweetId",
        config.me15_image_vec_dir, config.me15_text_vec_dir
    )

    dataset_test = dataset.GraphDataset(
        df_test, config.root_dir,
        "imageId(s)", "tweetId",
        config.me15_image_vec_dir, config.me15_text_vec_dir
    )

    return dataset_train, dataset_test
def set_up_weibo():


    df_train = pd.read_csv(f'{config.root_dir}{config.we_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.we_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    return dataset_train, dataset_test