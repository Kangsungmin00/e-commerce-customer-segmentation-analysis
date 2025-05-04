
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score

def merge_data(orders_df, order_items_df, customers_df, payments_df, products_df):
    customers_df = customers_df.drop_duplicates()
    merged_df = pd.merge(orders_df, customers_df, on='customer_id', how='left')
    merged_df = pd.merge(merged_df, order_items_df, on='order_id', how='left')
    merged_df = pd.merge(merged_df, payments_df, on='order_id', how='left')
    final_df = pd.merge(merged_df, products_df, on='product_id', how='left')
    return final_df

def clean_missing_values(df):
    df['product_weight_g'].fillna(df['product_weight_g'].median(), inplace=True)
    df['product_length_cm'].fillna(df['product_length_cm'].median(), inplace=True)
    df['product_height_cm'].fillna(df['product_height_cm'].median(), inplace=True)
    df['product_width_cm'].fillna(df['product_width_cm'].median(), inplace=True)
    
    columns_to_drop_null = ['order_approved_at', 'order_item_id', 'product_id', 'seller_id', 'price',
                            'shipping_charges', 'payment_sequential', 'payment_type',
                            'payment_installments', 'payment_value', 'product_category_name']
    df = df.dropna(subset=columns_to_drop_null)
    df = df.drop(columns=['payment_sequential'])
    
    df['order_item_id'] = df['order_item_id'].apply(lambda x: 1 if x == 1 else 0)
    
    def remove_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    for col in ['price', 'shipping_charges', 'payment_value',
                'product_weight_g', 'product_length_cm',
                'product_height_cm', 'product_width_cm']:
        df = remove_outliers_iqr(df, col)
    
    df = df[df['payment_installments'] <= 12]
    
    df['volume'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df = df.drop(columns=['product_length_cm', 'product_height_cm', 'product_width_cm'])
    
    df['repeat_order'] = df.groupby('customer_id')['order_id'].transform('nunique')
    df['total_price'] = df.groupby('customer_id')['price'].transform('sum')
    df['avg_price'] = df.groupby('customer_id')['price'].transform('mean')
    df['total_payment_value'] = df.groupby('customer_id')['payment_value'].transform('sum')
    df['avg_payment_value'] = df.groupby('customer_id')['payment_value'].transform('mean')
    
    numerical_cols = ['price', 'shipping_charges', 'payment_value', 'product_weight_g', 'volume']
    df[numerical_cols] = np.log1p(df[numerical_cols])
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def perform_clustering(df, n_clusters=3):
    cluster_features = ['repeat_order','total_price', 'avg_price', 'total_payment_value', 'avg_payment_value']
    cluster_df = df[cluster_features].copy()
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, random_state=42)
    df['cluster'] = kmeans.fit_predict(cluster_df)
    
    summary = df.groupby('cluster')[cluster_features].mean()
    return df, summary
