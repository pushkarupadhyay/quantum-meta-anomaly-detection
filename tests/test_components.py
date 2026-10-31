import unittest
import numpy as np
from src.datasets import DatasetManager
from src.pqc_encoder import PQCEncoder
from src.fewshot_head import FewShotHead
from src.meta_rl_es import EvolutionStrategies

class TestQuantumMetaRL(unittest.TestCase):
    def test_dataset_manager(self):
        dm = DatasetManager(n_qubits=4)
        dm.load_all()
        self.assertIn('breast_cancer', dm.datasets)
        X_s, y_s, X_q, y_q = dm.sample_episode('breast_cancer', k_support=5, q_query=10)
        self.assertEqual(X_s.shape, (10, 4))
        self.assertEqual(y_s.shape, (10,))
        self.assertEqual(X_q.shape, (10, 4))
        self.assertEqual(y_q.shape, (10,))

    def test_pqc_encoder(self):
        pqc = PQCEncoder(n_qubits=4, n_layers=1)
        x = np.random.uniform(-np.pi, np.pi, 4)
        output = pqc.forward(x)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(np.all(output >= -1.0) and np.all(output <= 1.0))

    def test_fewshot_head(self):
        head = FewShotHead()
        X_s = np.random.randn(10, 4)
        y_s = np.array([0,0,0,0,0, 1,1,1,1,1])
        head.fit(X_s, y_s)
        X_q = np.random.randn(5, 4)
        probas = head.predict_proba(X_q)
        self.assertEqual(probas.shape, (5,))

    def test_es_update(self):
        es = EvolutionStrategies(params_shape=(1, 4, 2), population_size=4)
        population, epsilons = es.sample_population()
        rewards = [0.1, 0.5, 0.2, 0.8]
        old_params = es.get_params().copy()
        es.update(rewards, epsilons)
        new_params = es.get_params()
        self.assertFalse(np.array_equal(old_params, new_params))

if __name__ == '__main__':
    unittest.main()
