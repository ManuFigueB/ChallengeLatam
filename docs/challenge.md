### Bug fixes 
- barplots had positional arguements that needed to be specified (added x= and y=)
- highseason function was ignoring the last day in each range (strptime gives 00:00:00, so with <= it was being left out, changed to next day and strict <)

### Model selection
Because the data is heavily unbalanced, a model with data balancing must be selected, otherwise most predictions say 0. The performance is similar between the balanced models, but xgboost has a very slight edge, also in my experience xgboost and boosting models in general tend to be more reliable and scalable.

The real input data for the model comes from just 3 columns: OPERA, TIPO_VUELO and MES.

### API creation
The model needs only 3 data points to create a predictions, so only those will need to be passed. Started with a very simple api, then adjusted it to pass the api-test, adding error verification for MES and TIPO_VUELO and adjusting input and output names/functions to pass expected tests here.