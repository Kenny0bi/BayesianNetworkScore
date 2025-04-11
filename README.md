# Bayesian Network Project Summary

## Project Overview

This project involves building a **Bayesian Network** to analyze relationships between variables in three datasets: small, medium, and large. The **K2 search algorithm** was used to learn the network structure, and the model was evaluated based on the **Bayesian score**. The objective was to find optimal configurations for each dataset by achieving the best possible Bayesian score through parameter tuning and data preprocessing.

---

## Small Dataset

### Dataset Details:
The small dataset `small.csv` contains the following variables:
- `age`, `portembarked`, `fare`, `numparentschildren`, `passengerclass`, `sex`, `numsiblings`, `survived`

### Approach:
1. **K2 Search Algorithm**: 
   - Experimented with different values for `max_parents` (2, 4, 5) and `node_order`.
   - Bayesian score calculated to evaluate the network structure.
   
2. **Best Results**:
   - **Best Configuration**: `max_parents=4`, `node_order=[7, 6, 5, 4, 3, 2, 1, 0]`
   - **Bayesian Score**: **-3798.45**

### Challenges:
- Adjustments were made to ensure correctness in Bayesian score calculation.
- Iterative experimentation to find the optimal network structure.

---

## Medium Dataset

### Dataset Details:
The medium dataset `medium.csv` contains variables such as `alcohol`, `fixedacidity`, and others. Binning was applied for more effective analysis.

### Approach:
1. **Data Preprocessing**:
   - `alcohol` and `fixedacidity` were binned into 5 categories.
   - Missing values were handled by filling with the mode.

2. **K2 Search Algorithm**:
   - Performed K2 search with various parameters: `max_parents`, `node_order`, and **ESS** (Equivalent Sample Size).
   
3. **Best Results**:
   - **Best Configuration**: `max_parents=4`, `node_order=[3, 1, 4, 0, 2, 6, 5, 7, 8, 9, 10, 11, 12]`, **ESS=1.0**
   - **Bayesian Score**: **-97,320**

### Challenges:
- Proper binning and handling missing data were critical preprocessing steps.

---

## Large Dataset

### Dataset Details:
The large dataset `large.csv` contains 50 variables, with unique values for each variable ranging from 2 to 4.

### Approach:
1. **Data Preprocessing**:
   - Handled missing values by filling them with the mode.
   
2. **K2 Search Algorithm**:
   - Ran K2 search with extensive parameter tuning to identify the optimal Bayesian network structure.
   
3. **Best Results**:
   - **Best Configuration**: `max_parents=2`, `ESS=2.0`
   - **Bayesian Score**: **-478,354.55**

### Challenges:
- The large number of variables increased the computational complexity and runtime, but through careful parameter tuning, the best results were achieved.

---

## Summary of Results

| Dataset        | Best Configuration                                                        | Bayesian Score  |
|----------------|---------------------------------------------------------------------------|-----------------|
| Small Dataset  | `max_parents=4`, `node_order=[7, 6, 5, 4, 3, 2, 1, 0]`                    | **-3798.45**    |
| Medium Dataset | `max_parents=4`, `node_order=[3, 1, 4, 0, 2, 6, 5, 7, 8, 9, 10, 11, 12]`, **ESS=1.0** | **-97,320**     |
| Large Dataset  | `max_parents=2`, `ESS=2.0`                                                | **-478,354.55** |

---

## Conclusion

This project successfully applied Bayesian Networks to model relationships across three datasets using the K2 search algorithm. Through detailed parameter tuning and data preprocessing, optimal Bayesian scores were achieved for each dataset. The project highlights the importance of iterative testing and configuration optimization in producing effective Bayesian Network models.

## 🙌 Author

**Kehinde Obidele**  
Graduate Student – Health Informatics  
Bayesian Networks | Probabilistic Modeling | Data-Driven Decision Support  

