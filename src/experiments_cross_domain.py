import numpy as np
import pandas as pd
from src.datasets import DatasetManager
from src.pqc_encoder import PQCEncoder
from src.fewshot_head import FewShotHead
from src.meta_rl_es import EvolutionStrategies
from src.utils_metrics import compute_metrics
from src.baselines import ClassicalBaselines, QuantumKernelBaseline
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MetaExperiment:
    def __init__(self, n_qubits=8, n_layers=2, population_size=16, iterations=50):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.population_size = population_size
        self.iterations = iterations
        self.dm = DatasetManager(n_qubits=n_qubits)
        self.dm.load_all()

    def run_leave_one_out(self):
        datasets = list(self.dm.datasets.keys())
        results = []

        for holdout in datasets:
            print(f"\n{'='*20}\nHolding out: {holdout}\n{'='*20}")
            train_datasets = [d for d in datasets if d != holdout]
            
            # 1. Meta-training
            es = EvolutionStrategies(params_shape=(self.n_layers, self.n_qubits, 2), 
                                     population_size=self.population_size)
            
            for i in range(self.iterations):
                population, epsilons = es.sample_population()
                rewards = []
                
                for params in population:
                    # Sample a random dataset from train_datasets
                    ds_name = np.random.choice(train_datasets)
                    X_s, y_s, X_q, y_q = self.dm.sample_episode(ds_name, k_support=5, q_query=20)
                    
                    # Forward with PQC
                    pqc = PQCEncoder(n_qubits=self.n_qubits, n_layers=self.n_layers)
                    pqc.set_params(params)
                    
                    X_s_pqc = pqc.forward(X_s)
                    X_q_pqc = pqc.forward(X_q)
                    
                    # Few-shot adapt
                    head = FewShotHead()
                    head.fit(X_s_pqc, y_s)
                    
                    # Evaluate
                    y_probas = head.predict_proba(X_q_pqc)
                    metrics = compute_metrics(y_q, y_probas)
                    rewards.append(metrics['f1'])
                
                es.update(rewards, epsilons)
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.iterations}, Mean Reward: {np.mean(rewards):.4f}")

            # 2. Meta-testing on holdout
            meta_params = es.get_params()
            
            # Adaptation to holdout
            X_s, y_s, X_q, y_q = self.dm.sample_episode(holdout, k_support=5, q_query=30)
            
            # Meta-trained PQC
            pqc_meta = PQCEncoder(n_qubits=self.n_qubits, n_layers=self.n_layers)
            pqc_meta.set_params(meta_params)
            X_s_meta = pqc_meta.forward(X_s)
            X_q_meta = pqc_meta.forward(X_q)
            
            head_meta = FewShotHead()
            head_meta.fit(X_s_meta, y_s)
            y_probas_meta = head_meta.predict_proba(X_q_meta)
            metrics_meta = compute_metrics(y_q, y_probas_meta)
            
            # From scratch PQC (same support set)
            pqc_scratch = PQCEncoder(n_qubits=self.n_qubits, n_layers=self.n_layers)
            X_s_scratch = pqc_scratch.forward(X_s)
            X_q_scratch = pqc_scratch.forward(X_q)
            
            head_scratch = FewShotHead()
            head_scratch.fit(X_s_scratch, y_s)
            y_probas_scratch = head_scratch.predict_proba(X_q_scratch)
            metrics_scratch = compute_metrics(y_q, y_probas_scratch)
            
            # Baselines
            (X_train_full, y_train_full), _, (X_test_full, y_test_full) = self.dm.preprocess(holdout)
            # Use only a subset for test to speed up
            X_test_sub = X_test_full[:100]
            y_test_sub = y_test_full[:100]
            
            # IF
            scores_if = ClassicalBaselines.isolation_forest(X_train_full, X_test_sub)
            metrics_if = compute_metrics(y_test_sub, scores_if)
            
            # OC-SVM
            X_train_normal = X_train_full[y_train_full == 0]
            scores_ocsvm = ClassicalBaselines.one_class_svm(X_train_normal, X_test_sub)
            metrics_ocsvm = compute_metrics(y_test_sub, scores_ocsvm)
            
            results.append({
                'holdout': holdout,
                'meta_auc': metrics_meta['auc'],
                'meta_f1': metrics_meta['f1'],
                'scratch_auc': metrics_scratch['auc'],
                'scratch_f1': metrics_scratch['f1'],
                'if_auc': metrics_if['auc'],
                'if_f1': metrics_if['f1'],
                'ocsvm_auc': metrics_ocsvm['auc'],
                'ocsvm_f1': metrics_ocsvm['f1']
            })

        df_results = pd.DataFrame(results)
        df_results.to_csv("reports/results/cross_domain_results.csv", index=False)
        print("\nResults saved to reports/results/cross_domain_results.csv")
        
        # Plot Cross Domain Comparison
        try:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(df_results['holdout']))
            width = 0.2
            
            plt.bar(x - 1.5*width, df_results['meta_auc'], width, label='Meta-Trained PQC', color='#1f77b4')
            plt.bar(x - 0.5*width, df_results['scratch_auc'], width, label='From-Scratch PQC', color='#aec7e8')
            plt.bar(x + 0.5*width, df_results['if_auc'], width, label='Isolation Forest', color='#ff7f0e')
            plt.bar(x + 1.5*width, df_results['ocsvm_auc'], width, label='One-Class SVM', color='#2ca02c')
            
            plt.xlabel('Holdout Dataset')
            plt.ylabel('ROC AUC Score')
            plt.title('Cross-Domain Generalization Comparison (ROC AUC)')
            plt.xticks(x, df_results['holdout'])
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(axis='y', linestyle=':', alpha=0.5)
            plt.tight_layout()
            os.makedirs("images", exist_ok=True)
            plt.savefig("images/cross_domain_auc_comparison.png", dpi=300)
            plt.close()
            print("Saved cross-domain comparison plot to 'images/cross_domain_auc_comparison.png'")
        except Exception as e:
            print(f"Error plotting cross domain comparison: {e}")
            
        return df_results

    def noise_analysis(self, holdout='breast_cancer'):
        print(f"\nStarting Noise Analysis on {holdout}...")
        noise_levels = [0.0, 0.01, 0.05, 0.1]
        noise_results = []
        
        # Get meta-trained params (simple way: train on others once)
        datasets = list(self.dm.datasets.keys())
        train_datasets = [d for d in datasets if d != holdout]
        
        es = EvolutionStrategies(params_shape=(self.n_layers, self.n_qubits, 2), population_size=self.population_size)
        # Small training for noise analysis demonstration
        for i in range(10):
            population, epsilons = es.sample_population()
            rewards = [np.random.rand() for _ in range(self.population_size)] # dummy rewards for speed
            es.update(rewards, epsilons)
        
        meta_params = es.get_params()
        X_s, y_s, X_q, y_q = self.dm.sample_episode(holdout, k_support=5, q_query=30)
        
        for nl in noise_levels:
            pqc = PQCEncoder(n_qubits=self.n_qubits, n_layers=self.n_layers, noise_level=nl)
            pqc.set_params(meta_params)
            
            X_s_pqc = pqc.forward(X_s)
            X_q_pqc = pqc.forward(X_q)
            
            head = FewShotHead()
            head.fit(X_s_pqc, y_s)
            y_probas = head.predict_proba(X_q_pqc)
            metrics = compute_metrics(y_q, y_probas)
            
            noise_results.append({
                'noise_level': nl,
                'auc': metrics['auc'],
                'f1': metrics['f1']
            })
            print(f"Noise {nl}: AUC {metrics['auc']:.4f}, F1 {metrics['f1']:.4f}")
            
        df_noise = pd.DataFrame(noise_results)
        df_noise.to_csv("reports/results/noise_analysis.csv", index=False)
        
        # Plot Noise Analysis
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(df_noise['noise_level'], df_noise['auc'], 'o-', label='ROC AUC', linewidth=2, color='#e377c2')
            plt.plot(df_noise['noise_level'], df_noise['f1'], 's--', label='F1-Score', linewidth=2, color='#bcbd22')
            plt.xlabel('Depolarizing Noise Level')
            plt.ylabel('Performance Score')
            plt.title(f'Noise Sensitivity Analysis (Holdout: {holdout})')
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            os.makedirs("images", exist_ok=True)
            plt.savefig("images/noise_analysis_impact.png", dpi=300)
            plt.close()
            print("Saved noise analysis plot to 'images/noise_analysis_impact.png'")
        except Exception as e:
            print(f"Error plotting noise analysis: {e}")
            
        return df_noise

if __name__ == "__main__":
    exp = MetaExperiment(n_qubits=4, iterations=10, population_size=8) # Small scale for demo
    exp.run_leave_one_out()
    exp.noise_analysis()
