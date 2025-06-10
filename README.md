# README

**Course:** Statistical Signal Processing and Learning  
**Department:** Computer Engineering & Informatics (CEID), University of Patras  
**Academic Year:** 2024/25  

---

## Exercise 1: System Identification

In this assignment, you will explore the problem of identifying an unknown digital filter from noisy observations. The objectives are:

- **Theoretical Analysis**  
  - Derive the optimal Wiener solution for a given FIR system.  
  - Analyze the convergence behavior of the Least Mean Squares (LMS) algorithm under the mean criterion, and establish bounds on the step size μ.  

- **Practical Implementation**  
  - Compute the Wiener filter coefficients using the provided dataset.  
  - Implement the LMS algorithm with various step sizes, compare its coefficient estimates against the Wiener solution, and visualize both the coefficient trajectories and learning curves.  
  - Extend your experiments to a time‐varying system by simulating:  
    - Smoothly changing coefficients over 1,000 samples.  
    - Abrupt coefficient changes at the halfway point.  
  - Aggregate and plot averaged learning curves over multiple Monte Carlo runs.

---

## Exercise 2: Federated Learning

This assignment introduces federated learning, a framework for collaborative model training without sharing raw data. You will:

- **Centralized Baseline**  
  - Train a neural network (e.g., MLP or simple CNN) on the MNIST dataset in a standard centralized setup.  
  - Evaluate how batch size and learning rate affect training/validation loss and test accuracy.

- **IID Federated Scenario**  
  - Partition MNIST data evenly and randomly (IID) across 10 client devices.  
  - Implement the Federated Averaging (FedAvg) algorithm, track convergence of training and validation metrics, and compare with the centralized baseline.

- **Non‑IID Federated Scenario**  
  - Create a non‑IID data split where each client holds samples from only two digit classes.  
  - Repeat the FedAvg experiments, record performance metrics, and analyze the impact of data heterogeneity by comparing IID vs. non‑IID results.

---


