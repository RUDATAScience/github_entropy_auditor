
# Algorithmic Opinion Dynamics & Informational Health Diagnostics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive suite of computational models, sociophysical simulations, and diagnostic tools designed to evaluate **"Informational Health"** and **"Epistemic Injustice"** in massive-scale digital societies.

By extending classical opinion dynamics models—such as the **Hegselmann-Krause (Bounded Confidence) model** and the **Galam (Discrete Spin) model**—this project mathematically bridges human social behavior with the internal architectures of modern Artificial Intelligence (AI). We demonstrate how the "Law of Large Numbers" paradoxically leads to the deterministic erasure of minority signals (the **"Signal Cliff"**), and how structural phase transitions occur during socio-political crises, leading to irreversible societal deformations.

Furthermore, this repository provides tools for **Electoral Forensics** and evaluates algorithmic vulnerabilities in LLMs and network analyses (e.g., Softmax underflow, Curse of Dimensionality).

## Key Theoretical Contributions

Our research identifies three distinct critical thresholds governing the collapse of informational diversity:

1. **Topological Threshold ($\epsilon_c \approx 0.2$ - HK Model)**: The limit of tolerance. Below this, society fractures into disconnected echo chambers (Critical Slowing Down).
2. **Statistical Threshold ($N_c \approx 10^4$ - Extended Galam Model)**: The "Signal Cliff". At massive data scales, opinion variance collapses ($\sigma^2 \propto 1/N$), causing minority signals to be irreversibly crushed as statistical noise.
3. **Plastic Deformation Threshold ($A_{crit} = 0.6$ - Integrated Crisis Model)**: The limit of societal resilience. Extreme external crises trigger a permanent restructuring of the probability space, generating a "Centrist Blackhole" (hysteresis) and irreversible divergence of susceptibility.

## Repository Structure & Modules

The repository consists of multiple independent Python scripts, categorized into three main research domains:

### Part 1: Opinion Dynamics & Phase Transitions (`main1.py` - `main10.py`)
These scripts simulate the macro-behavior of social systems using statistical mechanics.
* **`main1.py`, `main4.py`, `main7.py`**: Validates the "Signal Cliff" in the Galam model, demonstrating variance collapse, Shannon entropy decay, and proving theoretical scaling limits ($1/N$).
* **`main2.py`, `main3.py`, `main5.py`, `main6.py`**: Simulates political crises, "Social Break" (unfollow cascades), and the irreversible "Centrist Blackhole" (Plastic Deformation / Hysteresis) using an extended continuous opinion model.
* **`main8.py`, `main9.py`, `main10.py`**: Computes advanced statistical mechanics proofs of phase transitions across HK, Galam, and Integrated models. Calculates Critical Slowing Down, probability density function (PDF) limits, and the Divergence of Susceptibility.

### Part 2: Algorithmic Bias & AI Architectures (`main12.py` - `main19.py`)
These scripts demonstrate how modern data science architectures inherently execute epistemic injustice.
* **`main12.py`, `main18.py`, `main19.py`**: Simulates trillion-scale ($10^{12}$) NLP clustering, proving that global optimization destroys diversity. Introduces **"Hierarchical Micro-chunking"** to rescue rare semantic patterns via Large Deviation Theory.
* **`main13.py`**: Visualizes the resolution limits of Modularity (e.g., Louvain method) in massive network clustering, structurally burying minority communities.
* **`main14.py`**: Proves how numerical stabilization in the Softmax function pushes minority logits past the Float64 underflow limit ($10^{-308}$), causing computational "epistemic death."
* **`main15.py`**: Demonstrates the topological collapse of distance metrics in high-dimensional vector embeddings (Curse of Dimensionality).
* **`main16.py`**: Massive-scale minority cluster survival simulations (from 10M to 10B items).

### Part 3: Social Anomaly Detection & Electoral Forensics (`main20.py` - `main22.py`)
Tools for quantifying artificial peer pressure and fraud.
* **`main20.py`**: Calculates the mathematical impossibility of 100% unanimity in large groups without censorship, mapping Social ($10^{-9}$), Physical ($10^{-80}$), and Computational ($10^{-308}$) limits.
* **`main21.py`, `main22.py`**: Interactive Electoral Forensic tools. Utilizes Kullback-Leibler (KL) Divergence and Large Deviation Theory to compute the $\log_{10} P$ probability of vote shares, detecting structural anomalies and organized voting manipulation.

## Getting Started

### Requirements
* Python 3.8 or higher
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scipy`

### Installation
Clone the repository and install dependencies:
```bash
git clone [https://github.com/your-username/algorithmic-opinion-dynamics.git](https://github.com/your-username/algorithmic-opinion-dynamics.git)
cd algorithmic-opinion-dynamics
pip install -r requirements.txt

```

### Usage Example

Run any script directly. For example, to view the advanced proofs of phase transitions in opinion dynamics (Critical Slowing Down, PDF heatmap, etc.):

```bash
python main10.py

```

Each script automatically generates comprehensive visualizations (`.png` files) and structured datasets (`.csv` files), often compressing them into `.zip` archives for easy downloading in Jupyter/Colab environments.

## References & Literature

This repository serves as the computational framework for research on Informational Phase Transitions. Key theoretical foundations include:

1. Kawahata, Y. (2026). *Information Phase Transitions and Epistemic Injustice in Massive Data*. MDPI Entropy (Submitted).
2. Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: models, analysis, and simulation. *JASSS*, 5(3).
3. Galam, S. (2008). Sociophysics: A review of Galam models. *International Journal of Modern Physics C*, 19(03), 409-440.
4. Touchette, H. (2009). The large deviation approach to statistical mechanics. *Physics Reports*, 478(1-3), 1-69.
5. Ishii, A., & Kawahata, Y. (2019). Opinion Dynamics Theory for Analysis of Consensus Formation and Division of Opinion on the Internet. *arXiv:1812.11845*.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

```

```
