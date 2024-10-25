library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)
library(ggplot2)


# Load in Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

train_data$ACTION = as.factor(train_data$ACTION)
# Data Cleaning and Preprocessing
train_data[-which(names(train_data) == "ACTION")] <- lapply(train_data[-which(names(train_data) == "ACTION")], factor)

## EDA

for (feature in names(train_data)) {
  if (feature != "action") {  # Skip the "action" column if you want
    ggplot(train_data, aes_string(x = feature)) + 
      geom_bar(fill = "skyblue", color = "black") +
      theme_minimal() +
      labs(title = paste("Distribution of", feature), x = feature, y = "Count") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))  }
}


my_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(-ACTION, fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors())  # dummy variable encoding

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)


### Logistic Regression

logRegModel <- logistic_reg() %>% #Type of model3
  set_engine("glm") 

## Put into a workflow here

log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data=train_data)

preds <- predict(log_wf, new_data=test_data, type="prob")
kaggle_submission <- preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)  
  

## Write out the file
vroom_write(x=kaggle_submission, file="./LogRegPreds.csv", delim=",")

=


####### Penalized Logistic Regression #######################
library(tidymodels)
library(embed) 
library(vroom)
library(tidyverse)
library(ggplot2)


# Load in Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

test_data[-which(names(test_data) == "id")] <- lapply(test_data[-which(names(test_data) == "id")], factor)

my_mod <- logistic_reg(mixture= tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

my_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(-ACTION, fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_lencode_glm(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # dummy variable encoding
  step_normalize(all_nominal_predictors()) 

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over10
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3) ## L^2 total tuning possibilities
## Split data for CV
folds <- vfold_cv(train_data, v = 5, repeats=1)

## Run the CV
CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- amazon_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=train_data)

preds <- predict(final_wf, new_data=test_data, type="prob")

kaggle_submission <- preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)  
vroom_write(x=kaggle_submission, file="./PenRegPreds.csv", delim=",")


