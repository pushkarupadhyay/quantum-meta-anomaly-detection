import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add current path to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_quantum_anomaly_detector import run_agentic_pipeline

st.set_page_config(
    page_title="Quantum Meta-Learning Anomaly Detection Agent",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚛️ Quantum Meta-Learning Anomaly Detection Agent")
st.markdown("""
This application orchestrates a **Quantum Meta-Learning** pipeline using **LangGraph** and **LangChain** 
to perform few-shot anomaly detection on heterogeneous tabular datasets. 

It evaluates a Parameterized Quantum Circuit (PQC) trained via episodic reinforcement learning (Evolution Strategies) 
against classical and quantum baselines across five core research questions (RQs).
""")

# SIDEBAR CONTROLS
st.sidebar.header("⚙️ Pipeline Configuration")

holdout_id = st.sidebar.number_input("OpenML Holdout Dataset ID", min_value=1, value=182, step=1, 
                                     help="The ID of the dataset to hold out and evaluate on. (37: Diabetes, 1481: CTG, 182: Satimage)")

holdout_name = st.sidebar.text_input("Holdout Dataset Name", value="satimage", 
                                     help="Name tag for the holdout dataset.")

n_qubits = st.sidebar.slider("Number of Qubits (Width)", min_value=2, max_value=8, value=8, step=1,
                            help="Feature dimensionality matched via PCA.")

n_layers = st.sidebar.slider("PQC Layers (Depth)", min_value=1, max_value=4, value=2, step=1)

iterations = st.sidebar.slider("Meta-RL Training Iterations", min_value=2, max_value=50, value=15, step=1,
                               help="Epochs for Evolution Strategies optimization.")

pop_size = st.sidebar.slider("ES Population Size", min_value=4, max_value=16, value=8, step=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Agent Setup")
openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", 
                                  help="Provide to enable GPT-4 mini agentic report synthesis. If left blank, a rule-based analyzer compiles the final report.")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

run_button = st.sidebar.button("🚀 Run Agentic Pipeline", use_container_width=True)

# MAIN PAGE LAYOUT
if run_button:
    with st.spinner("Executing LangGraph Orchestrated Workflow... (Evaluating Quantum States and Baselines)"):
        # We hook stdout to capture prints from the pipeline nodes
        import io
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            results, report = run_agentic_pipeline(
                holdout_dataset_id=int(holdout_id), 
                holdout_name=holdout_name, 
                iterations=int(iterations)
            )
            pipeline_success = True
        except Exception as e:
            pipeline_success = False
            pipeline_error = str(e)
            
        sys.stdout = old_stdout
        log_output = buffer.getvalue()
        
    if not pipeline_success:
        st.error(f"Pipeline execution failed: {pipeline_error}")
    else:
        st.success("Workflow completed successfully!")
        
        # Tabs for logs, structured results, and the agent report
        tab_report, tab_results, tab_logs = st.tabs(["📄 Agentic Research Report", "📊 Benchmark & RQ Analysis", "💻 LangGraph Node Logs"])
        
        with tab_report:
            st.markdown(report)
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=report,
                file_name=f"agentic_research_report_{holdout_name}.md",
                mime="text/markdown"
            )
            
        with tab_results:
            st.header("Research Question Benchmarks")
            
            # Matplotlib Visualizations
            st.subheader("📈 Performance Visualization")
            
            subtab1, subtab2, subtab3 = st.tabs(["🚀 Core Experiments", "🧬 Advanced Analysis", "🔬 Qubit & Kernel Benchmarks"])
            
            with subtab1:
                col1, col2 = st.columns(2)
                with col1:
                    if os.path.exists("images/es_training_convergence.png"):
                        st.image("images/es_training_convergence.png", caption="ES Training Convergence Progress", use_container_width=True)
                    if os.path.exists("images/learning_curves.png"):
                        st.image("images/learning_curves.png", caption="Episodic Learning Curves (5 Seeds)", use_container_width=True)
                with col2:
                    if os.path.exists("images/rq3_baseline_comparison.png"):
                        st.image("images/rq3_baseline_comparison.png", caption="RQ3 Model Benchmarks", use_container_width=True)
                    if os.path.exists("images/rq4_noise_robustness.png"):
                        st.image("images/rq4_noise_robustness.png", caption="RQ4 Noise Robustness (Depolarizing Channel)", use_container_width=True)
                        
            with subtab2:
                col3, col4 = st.columns(2)
                with col3:
                    if os.path.exists("images/model_stability_analysis.png"):
                        st.image("images/model_stability_analysis.png", caption="Model Stability under Perturbations", use_container_width=True)
                    if os.path.exists("images/sanity_check_data_distribution.png"):
                        st.image("images/sanity_check_data_distribution.png", caption="Sanity Check: Satimage Feature Space (PCA)", use_container_width=True)
                with col4:
                    if os.path.exists("images/roc_satimage_pqc_baselines.png"):
                        st.image("images/roc_satimage_pqc_baselines.png", caption="FPR vs TPR ROC Curves (Satimage)", use_container_width=True)
                    if os.path.exists("images/satimage_auc_f1_comparison.png"):
                        st.image("images/satimage_auc_f1_comparison.png", caption="ROC AUC vs F1 Score Comparison", use_container_width=True)
                        
            with subtab3:
                col5, col6 = st.columns(2)
                with col5:
                    if os.path.exists("images/qubit_count_comparison.png"):
                        st.image("images/qubit_count_comparison.png", caption="Qubit Count Scaling Lift: 4 vs 8 Qubits", use_container_width=True)
                with col6:
                    if os.path.exists("images/qiskit_pennylane_comparison.png"):
                        st.image("images/qiskit_pennylane_comparison.png", caption="Qiskit vs Pennylane Z-Expectations Verification", use_container_width=True)
            
            st.markdown("---")
            
            # RQ1
            st.subheader("RQ1: Quantum vs Classical Encoder Ablation")
            st.markdown("Under the same episodic meta-learning procedure, does a PQC out-perform a classical neural network encoder?")
            df_rq1 = pd.DataFrame(results["RQ1"]).T
            st.dataframe(df_rq1, use_container_width=True)
            
            # RQ2
            st.subheader("RQ2: Cross-Domain Meta-Learning Adaptation")
            st.markdown("Does meta-training on N-1 datasets improve few-shot performance on the holdout compared to training from scratch?")
            df_rq2 = pd.DataFrame(results["RQ2"]).T
            st.dataframe(df_rq2, use_container_width=True)
            
            # RQ3
            st.subheader("RQ3: Comparison to Standard & Quantum Kernel Baselines")
            st.markdown("How does the meta-trained PQC compare to Isolation Forest, One-Class SVM, and Quantum Kernel One-Class SVM?")
            df_rq3 = pd.DataFrame(results["RQ3"]).T
            st.dataframe(df_rq3, use_container_width=True)
            
            # RQ4 - Noise
            st.subheader("RQ4: Noise Robustness (Depolarizing Channel)")
            st.markdown("Performance metrics evaluated on simulated NISQ hardware with physical depolarizing noise:")
            df_rq4 = pd.DataFrame(results["RQ4_Noise"]).T
            st.dataframe(df_rq4, use_container_width=True)
            
            # RQ4 - Resources
            st.subheader("RQ4: Quantum Resource Analysis")
            st.markdown("Approximate requirements for running the model on physical hardware:")
            res_data = results["RQ4_Resources"]
            df_res = pd.DataFrame({
                "Metric": list(res_data.keys()),
                "Value": list(res_data.values())
            })
            st.dataframe(df_res, use_container_width=True)
            
        with tab_logs:
            st.subheader("Workflow Node Execution Logs")
            st.text_area("Execution trace:", value=log_output, height=400, disabled=True)

else:
    st.info("Configure the parameters in the sidebar and click **🚀 Run Agentic Pipeline** to begin.")
    
    # Static project overview / instruction
    st.subheader("ℹ️ Project Proposal Summary")
    st.markdown("""
    This project evaluates **Quantum Meta-Learning** for few-shot anomaly detection on heterogeneous tabular datasets. 
    It leverages **episodic meta-reinforcement learning** (Evolution Strategies) to pre-train a Parameterized Quantum Circuit (PQC) 
    that serves as a high-dimensional feature extractor. 

    The entire process is orchestrated through **LangGraph** (managing state and control flow) and **LangChain** (handling 
    report analysis and LLM explanations).
    
    ### 🔬 Research Questions Covered:
    * **RQ1 (Quantum vs Classical)**: Comparison of representation learning under matching meta-learning algorithms.
    * **RQ2 (Cross-Domain Meta-Learning)**: Comparison of Meta-trained PQC versus From-Scratch PQC.
    * **RQ3 (Comparison to Baselines)**: Benchmarks against Isolation Forest, One-Class SVM, and Quantum Kernel One-Class SVM.
    * **RQ4 (Noise & Resources)**: Profile performance under Depolarizing Noise channels and log shot/depth requirements.
    * **RQ5 (Agentic Workflow)**: Analysis of LangGraph/LangChain orchestration for debugging, reproduction, and agentic reporting.
    """)
