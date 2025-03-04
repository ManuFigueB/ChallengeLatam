### Bug fixes 
- barplots had positional arguements that needed to be specified (added x= and y=)
- highseason function was ignoring the last day in each range (strptime gives 00:00:00, so with <= it was being left out, changed to next day and strict <)

### Model selection
Because the data is heavily unbalanced, a model with data balancing must be selected. The performance is similar between the balanced models, but xgboost has a very slight edge, also in my experience xgboost and boosting models in general tend to be more reliable and scalable.

