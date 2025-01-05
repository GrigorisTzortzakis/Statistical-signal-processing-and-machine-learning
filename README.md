# Statistical Signal Processing and Machine Learning - Exercise 1

This repository contains the solutions to **Exercise 1** of the course "Statistical Signal Processing and Machine Learning." The exercise focuses on system identification for Linear Time-Invariant (LTI) and Linear Time-Variant (LTV) systems using Wiener and LMS algorithms. The implementation is carried out in **MATLAB** and **Python**, and the results are visualized using provided plots.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [LTI System Identification](#lti-system-identification)  
   - [Wiener Filter (Question 1.2)](#wiener-filter-question-12)  
   - [LMS Algorithm (Question 1.3)](#lms-algorithm-question-13)  
3. [LTV System Identification](#ltv-system-identification)  
   - [Smooth and Abrupt Variations (Question 2.3)](#smooth-and-abrupt-variations-question-23)  
   - [Error Analysis over Multiple Realizations (Question 2.4)](#error-analysis-over-multiple-realizations-question-24)  
4. [Results](#results)  
5. [Setup and Execution](#setup-and-execution)  

---

## Introduction

This exercise aims to identify unknown systems using Wiener and LMS adaptive filters. Two types of systems are studied:
1. **Linear Time-Invariant (LTI) Systems**: Fixed impulse response.
2. **Linear Time-Variant (LTV) Systems**: Impulse response varies over time (smooth or abrupt variations).

The questions are solved step-by-step, and the results are plotted to visualize the performance of each approach.

---

## LTI System Identification

### System Description

The LTI system is represented as shown below:

![LTI System](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/LTI_System.png)

The system's output \( d(n) \) is modeled using a known input \( x(n) \) and an unknown system \( H(z) \). This unknown system is identified using two approaches:
1. Wiener filter (optimal solution).
2. LMS algorithm (adaptive solution).

---

### Wiener Filter (Question 1.2)

The Wiener filter is used to identify the unknown system \( H(z) \) based on the input \( x(n) \) and the desired output \( d(n) \). The filter minimizes the mean squared error (MSE) between the system output and the desired signal.

**Result:**
![Wiener Filter - Question 1.2](<img src="https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Queston1.2-Wiener.png" alt="Wiener Filter - Question 1.2" width="600px">)

---

### LMS Algorithm (Question 1.3)

The LMS algorithm is an iterative approach to minimize the error \( e(n) \) and adaptively adjust the filter weights.

#### Question 1.3.2 - LMS Filter Coefficients

The LMS algorithm is implemented with 4 coefficients, initialized to zero. The results are as follows:

**Result:**
![LMS - Question 1.3.2](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question1.3.2-LMS.png)

---

#### Question 1.3.3 - LMS with Different Step Sizes

The LMS algorithm is analyzed for different step sizes \( \mu \): 
\[ \mu = [0.001 \mu_{\text{max}}, 0.01 \mu_{\text{max}}, 0.1 \mu_{\text{max}}, 0.5 \mu_{\text{max}}] \]

**Result:**
![Different Step Sizes - Question 1.3.3](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question1.3.3-Different-Step-Sizes.png)

---

#### Question 1.3.3.1 - Learning Curve

The learning curve for the LMS algorithm with different step sizes is plotted:

**Result:**
![Learning Curve - Question 1.3.3.1](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question1.3.3.1-Learning-Curve.png)

---

## LTV System Identification

### System Description

The LTV system introduces time-varying behavior into the impulse response. The system's structure is shown below:

![LTV System](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/LTV_System.png)

Two types of variations are studied:
1. **Smooth Variation**: A continuous change in the impulse response.
2. **Abrupt Variation**: Sudden changes in the impulse response at predefined intervals.

---

### Smooth and Abrupt Variations (Question 2.3)

The weights and learning curves for both variations are analyzed.

#### Weight Estimation

The estimated weights for both smooth and abrupt variations are:

**Result:**

![Weights of Each System - Question 2.3](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question2.3-Weights-of-each-system.png)

#### Weight Evolution

The evolution of the weights over iterations is visualized for both variations:

**Result:**
![Weight Evolution - Question 2.3](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question2.3-Weight-evolution.png)

#### Learning Curve

The learning curves for both smooth and abrupt variations are:

**Result:**
![Learning Curve - Question 2.3](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question2.3-Learning-Curve.png)

---

### Error Analysis over Multiple Realizations (Question 2.4)

The LMS algorithm is run for 20 different realizations of the reference signal \( d(n) \), and the average squared error is calculated and plotted.

**Result:**
![Average Error - Question 2.4](https://github.com/GrigorisTzortzakis/Statistical-signal-processing-and-machine-learning/blob/main/Excersice%201/Pics/Question2.4-Average-Error.png)

---

## Results

1. **Wiener Filter** provides optimal weights but requires prior knowledge of the system statistics.
2. **LMS Algorithm** iteratively adapts to the unknown system and converges to the optimal solution.
3. For **LTV systems**, smooth variations allow faster convergence compared to abrupt variations.
4. Average error analysis over multiple realizations confirms the robustness of the LMS algorithm.

---

## Setup and Execution

### Prerequisites

- MATLAB or Python installed on your system.
- Python dependencies (if using Python):
  ```bash
  pip install numpy matplotlib
