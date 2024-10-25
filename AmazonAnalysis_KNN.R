library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)

# Load in Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")
test_data[-which(names(test_data) == "id")] <- lapply(test_data[-which(names(test_data) == "id")], factor)
train_data$ACTION = as.factor(train_data$ACTION)
train_data[-which(names(train_data) == "ACTION")] <- lapply(train_data[-which(names(train_data) == "ACTION")], factor)


## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>%  #set or tune
  set_mode("classification") %>%
  set_engine("kknn")

my_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(-ACTION, fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # dummy variable encoding
  step_normalize(all_nominal_predictors()) 

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Grid of values to tune over10
tuning_grid <- grid_regular(neighbors(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics= ) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

preds <- predict(final_wf, new_data=test_data, type="prob")

kaggle_submission <- preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)  
vroom_write(x=kaggle_submission, file="./KNNPreds.csv", delim=",")











preds <- predict(final_wf, new_data=test_data, type="prob")
