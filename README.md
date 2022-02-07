# The Search for Happiness: Exploring Correlations in Global Life Satisfaction and Development Data

This is the repository of the Data Literacy Project by Christian Eberle and Samuel Wörz.

## Abstract

We tried to predict national happiness, i.e. the self-evaluation of one’s life satisfactions on a scale from 1 to 10 as presented in the World Happiness Report, using linear regression. Our predictors are economical, environmental and population indicators for each country provided by the World Bank. Using both regularized and non-regularized regression, we show that these development indicators allow for reasonably accurate predictions of life satisfaction (Mean-squared erros of 0.33 when regularized, else 0.55). Inspecting the most prominent coefficients, we identify groups of development indicators that appear especially predictive of life satisfaction. Nonetheless, no small set of variables turns out to be crucial to a models performance. Furthermore, we devised two methods of data reduction to enable non-regularized regression. To maximally decrease the vast multicolinearity in the development data without jointly removing clustered indicators, we iteratively remove the most correlated indicator based on either correlation coefficient or Variance Influence Factor. We find that removing the most correlated indicators oftentimes yields no performance increase compared to removing random indicators, given our data, and can even be detrimental.


## Results

- Given development data, removing the most correlated variables is not necessarily the best choice when aiming for best performing least squares regression models. Highly correlated variables could correspond with highly influential ones directly, or represent highly influential hidden factors. Let's take for example the crucial yet not directly measured reliance of a country's population on agriculture to support itself, a key trademark of the world's poorest states. This factor is represented by the rural population, high employment in agriculture rates, high formal unemployment rates, contributing family workers rates, and more. But due to the sheer importance of this factor, one would worsen the model by removing some of its highly correlated but nevertheless informative indicators.

![](figures/reduction_plots.pdf) 

- The average life satisfaction in a state can be predicted fairly accurately based on development indicators. While the predictabilty does not rely heavily on single indicators, we were able to identify a group of indicators that are especially helpful for predicting life satisfacton. These appear to reflect characteristics of underdeveloped countries, such as lower life expectancy at birth or higher employment rates in agriculture. 
![](figures/ridge_coefs.pdf) 

## Download Links

Original Data:
- [World Bank Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
- [World Happinesss Report 2020](https://www.kaggle.com/mathurinache/world-happiness-report)


