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


### Improving the model

Abaixo serão apresentados diversos testes realizados na obtenção de features adicionais. Em cada teste, ele será considerando no modelo final se possuir resultado maior ou igual a obtido anteriomente no Kaggle.

**Teste 1:** todos os features extraído utilizando a DWT (seção Wavelets) foram obtidos utilizando a imagem original. Nesse teste, o mesmo processo foi realizado com o ruído da imagem, dobrando a quantidade de features (216 features). Treinando esse modelo obtém-se uma precisão de 87% e 39%, na validação e no Kaggle respectivamente. Esses novos features serão utilizados.

**Teste 2:** aplicação do LBP (Seção LBP) sobre a imagem original em tons de cinza: 86% na validação, 40.3% no Kaggle, 26 novos features. Aplicação do LBP no ruído em tons de cinza: 85% na validação e 41.27% no Kaggle, 26 novos features. Aplicação do LBP na imagem original e no ruído (ambas em tons de cinza): 87% na validação e 40.2% no Kaggle, 52 novos features. Nesse caso foi considerado o que obteve maior precisão no Kaggle.

**Teste 3:** no teste anterior foi obtida a melhor precisão no Kaggle quando aplicando o LBP somente no ruído em tons de cinza, então uma possibilidade de melhorar é fazer a aplicação do LBP por canal do ruído. Feito isso, tempos: 85% na validação, 47% no Kaggle, 78 novos features. Esses features foram considerando substituindo os features do Teste 2 e totalizando 294 features.

**Teste 4:** neste teste foi aplicado o LBP para cada vetor de coeficiente da DTW (seção Wavelets), obtendo: 82% na validação, 40% no Kaggle e 702 novos features. Como o resultado foi inferior ao do Teste 3, estes novos features foram descartados.

**Teste 5:** no Teste 3, cada aplicação do LBP extrai 26 features, conforme mencionado na seção LBP. O Teste 5 consiste em não montar o histograma no LBP, e apenas extrair as estatísticas na imagem retornada no LBP, ou seja, cada aplicação do LBP agora irá extrair 4 features ( mean, variance, skew e kurtosis). Com isso o total de features foi reduzido para 228, obtendo 86.7% de precisão na validação e 47% de precisão no Kaggle. Houve um pequeno aumento na precisão da validação em relação ao ultimo teste considerado (Teste 3), logo esse teste foi considerados.

**Teste 6:** o teste 4 foi descartado por não aumentar a precisão, no entanto, no intuito de inclui-lo no modelo, foi realizado o mesmo procedimento do Teste 5. A precisão obtida foi 87.45% na validação  e 44% no Kaggle. Mais uma vez, o teste foi descartado.

**Teste 7:** o ultimo teste consistiu em adicionar a extração de uma nova estatística no conjunto de funções que extrair as estatística de uma imagem/vetor considerados neste trabalho. Foi adicionar o calculo do desvio padrão, essa alteração obteve 89.45% na validação e 50.85% no Kaggle, sendo este o ultimo teste realizando na extração de features da imagem, com um total de 348 features.


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
