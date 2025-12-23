# List of cell-level features

## First Order Features

1. Energy
   $$
   \text{Energy} = \sum_{i=1}^{N_p} \left( X(i) + c \right)^2
   $$
   Measures the magnitude of voxel intensities. Larger values indicate higher overall intensity.

2. Total Energy
   $$
   \text{Total Energy} = V_{\text{voxel}} \sum_{i=1}^{N_p} \left( X(i) + c \right)^2
   $$
   Energy scaled by voxel volume $V_{\text{voxel}} (mm^3)$ .

3. Entropy
   $$
   \text{Entropy} = -\sum_{i=1}^{N_g} p(i)\log_2 \left( p(i) + \varepsilon \right)
   $$
   Measures randomness or uncertainty in intensity distribution.

4. Minimum
   $$
   \text{Minimum} = \min(X)
   $$
   Lowest gray-level intensity in the ROI.

5. 10th percentile
   $$
   P_{10} = \text{10th percentile of } X
   $$

6. 90th percentile
   $$
   P_{90} = \text{90th percentile of } X
   $$

7. Maximum
   $$
   \text{Maximum} = \max(X)
   $$
   Highest gray-level intensity in the ROI.

8. Mean
   $$
   \bar{X} = \frac{1}{N_p}\sum_{i=1}^{N_p} X(i)
   $$
   Average gray-level intensity.

9. Median
   $$
   \text{Median} = \text{50th percentile of } X
   $$

10. Interquartile Range
    $$
    \text{IQR} = P_{75} - P_{25}
    $$
    Difference between the 75th and 25th percentiles.

11. Range
    $$
    \text{Range} = \max(X) - \min(X)
    $$
    Difference between maximum and minimum intensities.

12. Mean Absolute Deviation (MAD)
    $$
    \text{MAD} = \frac{1}{N_p} \sum_{i=1}^{N_p} \left| X(i) - \bar{X} \right|
    $$
    Mean distance of intensities from the mean.

13. Robust Mean Absolute Deviation (rMAD)
    $$
    \text{rMAD} = \frac{1}{N_{10-90}} \sum_{i=1}^{N_{10-90}} \left| X_{10-90}(i) - \bar{X}_{10-90} \right|
    $$
    Computed using only intensities between the 10th and 90th percentiles.

14. Root Mean Squared (RMS)
    $$
    \text{RMS} = \sqrt{ \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) + c \right)^2 }
    $$
    Square root of the mean of squared intensities.

15. Standard Deviation
    $$
    \sigma = \sqrt{ \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^2 }
    $$
    Measures dispersion around the mean.

16. Skewness
    $$
    \text{Skewness} =
    \frac{ \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^3 }
    { \left( \sqrt{ \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^2 } \right)^3 }
    $$
    Measures asymmetry of the intensity distribution.

17. Kurtosis
    $$
    \text{Kurtosis} =
    \frac{ \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^4 }
    { \left( \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^2 \right)^2 }
    $$
    Measures peakedness of the distribution.
    (Note: IBSI uses excess kurtosis, subtracting 3.)

18. Variance
    $$
    \text{Variance} = \frac{1}{N_p} \sum_{i=1}^{N_p} \left( X(i) - \bar{X} \right)^2
    $$
    Mean squared deviation from the mean.

19. Uniformity
    $$
    \text{Uniformity} = \sum_{i=1}^{N_g} p(i)^2
    $$
    Measures homogeneity of the intensity distribution.

## Shape Features

Let the ROI be represented by a surface or perimeter mesh with $N_f$ elements, voxel spacing taken into account where applicable.

1. Mesh Surface
   $$
   A_i = \frac{1}{2} \left| \vec{Oa_i} \times \vec{Ob_i} \right|\\ A = \sum_{i=1}^{N_f} A_i
   $$
   Where $\vec{Oa_i}$ and $\vec{Ob_i}$ are edge vectors of the i-th triangle in the mesh.
   Measures the exact surface area of the ROI using a triangulated mesh representation.

2. Pixel Surface
   $$
   A_{\text{pixel}} = \sum_{k=1}^{N_v} A_k
   $$
   Approximates the surface area by multiplying the number of pixels by the area of a single pixel.
   This method does **not** use the mesh and is less precise.

3. Perimeter
   $$
   P_i = \sqrt{(a_i - b_i)^2}\\ P = \sum_{i=1}^{N_f} P_i
   $$
   Where $a_i$ and $b_i$ are vertices of the i-th perimeter segment.
   Represents the total boundary length of the ROI.

4. Perimeter to Surface Ratio
   $$
   \text{Perimeter-to-Surface Ratio} = \frac{P}{A}
   $$
   Lower values indicate more compact, circle-like shapes
   This feature is **not dimensionless**.

5. Sphericity
   $$
   \text{Sphericity} = \frac{2\pi \sqrt{A/\pi}}{P}
   $$
   Measures how close the ROI shape is to a perfect circle.
   Values range from $0 < \text{Sphericity} \leq 1$ , where 1 indicates a perfect circle.

6. Spherical Disproportion
   $$
   \text{Spherical Disproportion} = \frac{P}{2\pi \sqrt{A/\pi}}
   $$
   Inverse of Sphericity.
   Values are $\geq$ 1, where 1 indicates a perfect circle.

7. Maximum Diameter
   $$
   \text{Maximum Diameter} = \max_{i,j} \left\| v_i - v_j \right\|
   $$
   Defined as the maximum Euclidean distance between any two surface mesh vertices.

8. Major Axis Length
   $$
   \text{Major Axis Length} = 4\sqrt{\lambda_{\text{major}}}
   $$
   Where $\lambda_{\text{major}}$ is the largest eigenvalue from PCA of the ROI voxel coordinates.
   Represents the longest axis of the ROI-enclosing ellipsoid.

9. Minor Axis Length
   $$
   \text{Minor Axis Length} = 4\sqrt{\lambda_{\text{minor}}}
   $$
   Where $\lambda_{\text{minor}}$ is the second-largest PCA eigenvalue.
   Represents the second-largest axis of the ROI-enclosing ellipsoid.

10. Elongation
    $$
    \text{Elongation} = \sqrt{\frac{\lambda_{\text{minor}}}{\lambda_{\text{major}}}}
    $$
    Measures how elongated the shape is.
    Values range from 0 (maximally elongated) to 1 (circle-like).

11. Mean Curvature
    $$
    H = \frac{1}{2}(k_1 + k_2)
    $$
    Here, k_1 and k_2 are the **principal curvatures** at a surface point (the maximum and minimum normal curvatures).

12. Gaussian Curvature
    $$
    K = k_1 \cdot k_2
    $$
    Measures intrinsic surface curvature, independent of embedding.

13. Maximum Curvature
    $$
    k_{\max} = \max(k_1, k_2)
    $$
    Represents the strongest local bending direction.

14. Minimum Curvature
    $$
    k_{\min} = \min(k_1, k_2)
    $$
    Represents the weakest local bending direction.

## Gray Level Co-occurrence Matrix (GLCM) Features

To be supplemented.

## Gray Level Size Zone Matrix (GLSZM) Features

To be supplemented.

## Gray Level Run Length Matrix (GLRLM) Features

To be supplemented.

## Neighbouring Gray Tone Difference Matrix (NGTDM) Features

To be supplemented.

## Gray Level Dependence Matrix (GLDM) Features

To be supplemented.