library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)

# read in data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")
test_data[-which(names(test_data) == "id")] <- lapply(test_data[-which(names(test_data) == "id")], factor)
train_data$ACTION = as.factor(train_data$ACTION)
train_data[-which(names(train_data) == "ACTION")] <- lapply(train_data[-which(names(train_data) == "ACTION")], factor)


# write recipe
my_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(-ACTION, fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # dummy variable encoding
  step_normalize(all_nominal_predictors()) 

# define model
forest_mod <- rand_forest(mtry = tune(), 
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# create workflow
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

# set up grid of tuning values
forest_tuning_params <- grid_regular(mtry(range = c(1,10)),
                                     min_n(),
                                     levels = 5)
# set up k-fold CV
folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=forest_tuning_params,
            metrics=metric_set(roc_auc))

# find best tuning params
bestTuneForest <- CV_results %>%
  select_best(metric = "rmse")



# finalize workflow and make predictions
forest_model <- rand_forest(mtry = 10, 
                            min_n = 2,
                            trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_model) %>%
  fit(data=bike_train)

forest_preds <- predict(forest_wf, new_data=bike_test)

kaggle_submission <- forest_preds %>%
  bind_cols(., bike_test) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count = exp(count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

vroom_write(x=kaggle_submission, file="./ForestPreds.csv", delim=",")