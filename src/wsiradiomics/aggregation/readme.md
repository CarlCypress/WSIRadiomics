## Aggregation Functions

Let $f_j(i)$ denote the value of the j-th cell-level feature for the i-th cell, and let $N_c$ be the total number of cells.

1. Mean
   $$
   \mu_j = \frac{1}{N_c} \sum_{i=1}^{N_c} f_j(i)
   $$
   Captures the average value of the j-th feature across all cells.

2. Median
   $$
   \text{Median}_j = \text{median}\!\left( f_j(1), \dots, f_j(N_c) \right)
   $$
   Provides a robust estimate of the central tendency and is less sensitive to outliers.

3. Standard Deviation
   $$
   \sigma_j = \sqrt{ \frac{1}{N_c - 1} \sum_{i=1}^{N_c} \left( f_j(i) - \mu_j \right)^2 }
   $$
   Measures inter-cell variability of the feature.

4. Interquartile Range (IQR)
   $$
   \text{IQR}_j = Q_{0.75}(f_j) - Q_{0.25}(f_j)
   $$
   Quantifies dispersion using the difference between the 75th and 25th percentiles and is robust to extreme values.

5. Skewness
   $$
   \text{Skewness}_j =
   \frac{\sqrt{N_c(N_c-1)}}{N_c-2}
   \cdot
   \frac{
   \frac{1}{N_c} \sum_{i=1}^{N_c} \left( f_j(i) - \mu_j \right)^3
   }{
   \left(
   \frac{1}{N_c} \sum_{i=1}^{N_c} \left( f_j(i) - \mu_j \right)^2
   \right)^{3/2}
   }
   \quad (N_c \ge 3)
   $$
   Measures the asymmetry of the cell-level feature distribution.

6. Kurtosis
   $$
   \text{Excess Kurtosis}_j =\frac{N_c - 1}{(N_c - 2)(N_c - 3)}\left[(N_c + 1)\left(\frac{\frac{1}{N_c} \sum_{i=1}^{N_c} \left( f_j(i) - \mu_j \right)^4}{\left(\frac{1}{N_c} \sum_{i=1}^{N_c} \left( f_j(i) - \mu_j \right)^2\right)^2}-3\right)+6\right]\quad (N_c \ge 4)
   $$
   Measures the peakedness of the distribution relative to a normal distribution.
