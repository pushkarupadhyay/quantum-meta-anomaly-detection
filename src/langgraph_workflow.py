from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import numpy as np
from src.datasets import DatasetManager
from src.pqc_encoder import PQCEncoder
from src.fewshot_head import FewShotHead
from src.meta_rl_es import EvolutionStrategies
from src.utils_metrics import compute_metrics

class GraphState(TypedDict):
    iteration: int
    max_iterations: int
    params: np.ndarray
    dm: DatasetManager
    current_rewards: List[float]
    current_epsilons: List[np.ndarray]
    n_qubits: int
    n_layers: int
    population_size: int
    history: List[float]

def load_data_node(state: GraphState):
    dm = DatasetManager(n_qubits=state['n_qubits'])
    dm.load_all()
    return {"dm": dm, "iteration": 0}

def sample_and_evaluate_node(state: GraphState):
    dm = state['dm']
    params_base = state['params']
    population_size = state['population_size']
    
    # Simple ES step inside one node for clarity in graph
    epsilons = [np.random.randn(*params_base.shape) for _ in range(population_size)]
    rewards = []
    
    train_datasets = [d for d in dm.datasets.keys()]
    
    for eps in epsilons:
        params = params_base + 0.1 * eps # sigma = 0.1
        ds_name = np.random.choice(train_datasets)
        X_s, y_s, X_q, y_q = dm.sample_episode(ds_name)
        
        pqc = PQCEncoder(n_qubits=state['n_qubits'], n_layers=state['n_layers'])
        pqc.set_params(params)
        
        X_s_pqc = pqc.forward(X_s)
        X_q_pqc = pqc.forward(X_q)
        
        head = FewShotHead()
        head.fit(X_s_pqc, y_s)
        y_probas = head.predict_proba(X_q_pqc)
        metrics = compute_metrics(y_q, y_probas)
        rewards.append(metrics['f1'])
        
    return {"current_rewards": rewards, "current_epsilons": epsilons}

def update_params_node(state: GraphState):
    rewards = np.array(state['current_rewards'])
    epsilons = state['current_epsilons']
    params = state['params']
    
    # Standardize rewards
    if np.std(rewards) > 0:
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
    
    gradient = np.zeros_like(params)
    for i in range(len(rewards)):
        gradient += rewards[i] * epsilons[i]
    
    # lr = 0.2, sigma = 0.1
    new_params = params + (0.2 / (len(rewards) * 0.1)) * gradient
    
    return {
        "params": new_params, 
        "iteration": state['iteration'] + 1,
        "history": state.get('history', []) + [float(np.mean(state['current_rewards']))]
    }

def check_convergence_condition(state: GraphState):
    if state['iteration'] >= state['max_iterations']:
        return "evaluate"
    else:
        return "continue"

def evaluation_node(state: GraphState):
    print(f"Final evaluation after {state['iteration']} iterations...")
    # Add final evaluation logic here if needed
    return state

def create_workflow(n_qubits=4, n_layers=1, population_size=8, max_iterations=10):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("sample_and_evaluate", sample_and_evaluate_node)
    workflow.add_node("update_params", update_params_node)
    workflow.add_node("evaluate", evaluation_node)
    
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "sample_and_evaluate")
    workflow.add_edge("sample_and_evaluate", "update_params")
    
    workflow.add_conditional_edges(
        "update_params",
        check_convergence_condition,
        {
            "continue": "sample_and_evaluate",
            "evaluate": "evaluate"
        }
    )
    
    workflow.add_edge("evaluate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    initial_params = np.random.uniform(-np.pi, np.pi, size=(1, 4, 2))
    app = create_workflow(n_qubits=4, n_layers=1, population_size=4, max_iterations=5)
    
    final_state = app.invoke({
        "iteration": 0,
        "max_iterations": 5,
        "params": initial_params,
        "n_qubits": 4,
        "n_layers": 1,
        "population_size": 4,
        "history": []
    })
    
    print("Workflow finished. Mean reward history:", final_state['history'])
