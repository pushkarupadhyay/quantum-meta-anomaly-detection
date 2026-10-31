import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_breast_cancer_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # In breast cancer, 0 is malignant (anomaly), 1 is benign (normal)
    # We want 0 for normal, 1 for anomaly
    df['target'] = 1 - data.target
    return df

def load_ionosphere_dataset():
    # Ionosphere dataset on OpenML ID 59
    data = fetch_openml(data_id=59, as_frame=True, parser='auto')
    df = data.frame
    # target: 'g' (good) -> normal (0), 'b' (bad) -> anomaly (1)
    df['target'] = df['class'].apply(lambda x: 0 if x == 'g' else 1)
    df = df.drop(columns=['class'])
    return df

def load_credit_card_dataset():
    # Credit Card Fraud on OpenML ID 1597
    data = fetch_openml(data_id=1597, as_frame=True, parser='auto')
    df = data.frame
    # target: '0' -> normal, '1' -> anomaly
    df['target'] = df['Class'].astype(int)
    df = df.drop(columns=['Class'])
    return df

def load_nsl_kdd_dataset():
    # NSL-KDD on OpenML ID 40685
    data = fetch_openml(data_id=40685, as_frame=True, parser='auto')
    df = data.frame
    # target: '1' -> normal (0), anything else -> 1
    # In some versions it might be string, in others numeric
    df['target'] = df['class'].apply(lambda x: 0 if str(x) == '1' else 1)
    df = df.drop(columns=['class'])
    # NSL-KDD has categorical features, let's encode them
    # Select only numeric for simplicity or encode
    df_numeric = df.select_dtypes(include=[np.number])
    return df_numeric

class DatasetManager:
    def __init__(self, n_qubits=8):
        self.n_qubits = n_qubits
        self.datasets = {}
        self.scalers = {}
        self.pcas = {}

    def load_all(self):
        print("Loading Breast Cancer dataset...")
        self.datasets['breast_cancer'] = load_breast_cancer_dataset()
        print("Loading Ionosphere dataset...")
        self.datasets['ionosphere'] = load_ionosphere_dataset()
        print("Loading Credit Card dataset...")
        self.datasets['credit_card'] = load_credit_card_dataset()
        print("Loading NSL-KDD dataset...")
        self.datasets['nsl_kdd'] = load_nsl_kdd_dataset()

    def preprocess(self, name):
        df = self.datasets[name]
        X = df.drop(columns=['target']).values
        y = df['target'].values

        # Split into train/val/test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        self.scalers[name] = scaler

        # PCA to n_qubits
        pca = PCA(n_components=self.n_qubits)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        self.pcas[name] = pca

        # Normalize to [-pi, pi] for angle encoding
        def normalize_features(data):
            # Map to [-pi, pi]
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            # Avoid division by zero
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return 2 * np.pi * (data - min_val) / range_val - np.pi

        X_train = normalize_features(X_train)
        X_val = normalize_features(X_val)
        X_test = normalize_features(X_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def sample_episode(self, name, k_support=5, q_query=20):
        (X_train, y_train), _, _ = self.preprocess(name)
        
        # Sample k normals and k anomalies for support set
        normal_idx = np.where(y_train == 0)[0]
        anomaly_idx = np.where(y_train == 1)[0]
        
        support_normal_idx = np.random.choice(normal_idx, k_support, replace=False)
        support_anomaly_idx = np.random.choice(anomaly_idx, k_support, replace=False)
        
        X_support = np.concatenate([X_train[support_normal_idx], X_train[support_anomaly_idx]])
        y_support = np.concatenate([np.zeros(k_support), np.ones(k_support)])
        
        # Sample query set (disjoint from support)
        remaining_normal_idx = np.setdiff1d(normal_idx, support_normal_idx)
        remaining_anomaly_idx = np.setdiff1d(anomaly_idx, support_anomaly_idx)
        
        query_normal_idx = np.random.choice(remaining_normal_idx, q_query // 2, replace=False)
        query_anomaly_idx = np.random.choice(remaining_anomaly_idx, q_query // 2, replace=False)
        
        X_query = np.concatenate([X_train[query_normal_idx], X_train[query_anomaly_idx]])
        y_query = np.concatenate([np.zeros(q_query // 2), np.ones(q_query // 2)])
        
        return X_support, y_support, X_query, y_query

if __name__ == "__main__":
    manager = DatasetManager(n_qubits=8)
    manager.load_all()
    for name in manager.datasets:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = manager.preprocess(name)
        print(f"Dataset: {name}")
        print(f"  Train shape: {X_train.shape}, Anomaly rate: {np.mean(y_train):.4f}")
        print(f"  Test shape: {X_test.shape}, Anomaly rate: {np.mean(y_test):.4f}")
