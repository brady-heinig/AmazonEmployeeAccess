library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)
library(discrim)
library(naivebayes)

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

## nb model3
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
# set up grid of tuning values
nb_tuning_params <- grid_regular(Laplace(),
                                     smoothness(),
                                     levels = 5)
# set up k-fold CV
folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_params,
            metrics=metric_set(roc_auc))

# find best tuning params
bestTuneNB <- CV_results %>%
  select_best(metric = "roc_auc")



# finalize workflow and make predictions
nb_model <- naive_Bayes(Laplace=bestTuneNB$Laplace, smoothness=bestTuneNB$smoothness) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) %>%
  fit(data=train_data)

nb_preds <- predict(nb_wf, new_data=test_data, type = "prob")

kaggle_submission <- nb_preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(ACTION=.pred_class)  

vroom_write(x=kaggle_submission, file="./NBPreds.csv", delim=",")
