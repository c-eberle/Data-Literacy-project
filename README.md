# The Search for Happiness: Exploring Correlations in Global Life Satisfaction and Development Data

This is the repository of the Data Literacy Project by Christian Eberle and Samuel Wörz.

## Abstract

We tried to predict national happiness, i.e. the self-evaluation of one’s life satisfactions on a scale from 1 to 10 as presented in the World Happiness Report, using linear regression. Our predictors are economical, environmental and population indicators for each country provided by the World Bank. Using both regularized and non-regularized regression, we show that these development indicators allow for reasonably accurate predictions of life satisfaction (Mean-squared erros of 0.33 when regularized, else 0.55). Inspecting the most prominent coefficients, we identify groups of development indicators that appear especially predictive of life satisfaction. Nonetheless, no small set of variables turns out to be crucial to a models performance. Furthermore, we devised two methods of data reduction to enable non-regularized regression. To maximally decrease the vast multicolinearity in the development data without jointly removing clustered indicators, we iteratively remove the most correlated indicator based on either correlation coefficient or Variance Influence Factor. We find that removing the most correlated indicators oftentimes yields no performance increase compared to removing random indicators, given our data, and can even be detrimental.

## Download Links

Original Data:
- [World Bank Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
- [World Happinesss Report 2020](https://www.kaggle.com/mathurinache/world-happiness-report)


