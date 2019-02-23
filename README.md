# Camera Model Identification
For more information, read Report.pdf (in Portuguese) or contact me.

## Activities
* Loading, modifying and expanding the dataset;
* Feature extraction and Feature scaling;
* Exploration Logistic Regression (LogReg) and Neural Network (NN).

## The dataset
Camera Model Identification. Contains 2750 images for training and 2640 for testing.

## Feature Extration
### Local Binary Pattern

24 points and ray 8.

### Noise extraction
With openCV: fastNlMeansDenoisingColored.

### Wavelets
* Decomposition multi level utilizando Discrete Wavelet Transform (DWT);
* Inverse Discrete Wavelet Transform (IDWT);
* Wiener filter;
* histogram, mean, variance, skew e kurtosis.

### Scaling
 * Rescaling;
 * Mean normalisation;
 * Standardization.

### Test on Kaggle

Val: 92.5%
Test: 23%

### Data Augmentation
* JPEG compression with quality factor = 70;
* JPEG compression with quality factor = 90;
* Resizing (factor of 0.5);
* Resizing (factor of 0.8);
* Resizing (factor of 1.5);
* Resizing (factor of 2.0);
* Gamma correction (gamma = 0.8);
* Gamma correction (gamma = 1,2);
* Rotation of 90;
* Rotation of 180;
* Rotation of 270;

Amount of data: 30250

| Train  | Val  | Teste (Kaggle) |
--- | --- | --- |
| train 1      | 80%                           | 36%                               | 
| train 2      | 38%                           | 23%                               | 
| train 3      | 58%                           | 33%                               | 
| train 4      | 40%                           | 27%                               | 

</p>

### Improving the model
**Test 1:** Val:87% , Kaggle: 39%.

**Test 2:** Val:87% , Kaggle: 40%.

**Test 3:** Val:85% , Kaggle: 47%.

**Test 4:** Val:82% , Kaggle: 40%.

**Test 5:** Val:86% , Kaggle: 47%.

**Test 6:** Val:87% , Kaggle: 44%.

**Test 7:** Val:89% , Kaggle: 51%.


## Model complexity

| Model       | Val  | Kaggle   | Scale         | Reg |
| -------------|------------|-----------|--------------|----------------------|
| -            | 89.63%    | 47.00%  | MN            | l2            |
| 10^2       | 88.54%    | 49.56%  | ST            | l2            |
| 10^2       | 91.45%    | 48.00%  | MN            | l2            |
| 10^2       | 88.00%    | 49.00%  | MN+ ST        | l2            |
| 10^2       | 89.81%    | 47.00%  | MN + ST       | -             |
| 10^3       | 87.09%    | 49.43%  | ST            | l2            |
| 10^3       | 87.00%    | 49.00%  | MN+ ST        | l2            |
| 10^4       | 40.00%    | 47.00%  | MN            | -             |
| 10^5       | 90.00%    | 47.00%  | MN            | -             |
| 10^100   | 87.00%    | 49.37%  | MN + ST       | l2            |
| 10^100   | 89.27%    | 50.58%  | MN + ST       | -             |
| 10^100   | 89.27%    | 51.06%  | ST + MN + ST  | l2            |


## Increased Dataset and Training with Probability

| Train  |Val  |Teste (Kaggle) |
| ------------|---------|-------|
| train 1      | 89.27%            | 51.06%                               |
| train 2      | 70.89%            | 42.31%                               |
| train 3      | 80.17%            | 49.93%                               |
| train 4      | 71.54%            | 44.04%                               |


| Model  |Val  |Kaggle |
| ---------|----------|----------------|
| Sem aumento de complexidade      | 74.50%            | 44.47%                               |
| Com aumento de complexidade      | 73.45%            | 43.22%                               |


## Neural Network

<p align="center">
  <img src="imgs/layers.png">
</p>
 

| Model  |Layers   | Val  | Teste (Kaggle)   |
| ---------|---------|----|------|
| 1       | 200                          | 93.64%                        | 49.08%                              |
| 1       | 875                          | 93.82%                        | 50.43%                              |
| 2       | 500                          | 93.45%                        | 49.16%                              |
| 2       | 650                          | 93.81%                        | 50.49%                              |
| 3       | 275                          | 80.18%                        | 47.30%                              |
| 3       | 800                          | 80.00%                        | 47.70%                              |


## Summary of Results

<p align="center">
  <img src="imgs/report.png">
</p>

| Classifier   | Val  | Kaggle  | 
| -------|------|--------|
| LogR                                | 91.45%                        | 51.06%                              |
| Perceptron                          | 93.82%                        | 50.49%                              |
