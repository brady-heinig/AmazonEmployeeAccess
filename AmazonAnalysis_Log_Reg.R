library(tidymodels)
library(vroom)
library(tidyverse)
library(ggplot2)


# Load in Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

train_data$ACTION = as.factor(train_data$ACTION)
train_data[-which(names(train_data) == "ACTION")] <- lapply(train_data[-which(names(train_data) == "ACTION")], factor)
# Data Cleaning and Preprocessing

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

test_data[-which(names(test_data) == "id")] <- lapply(test_data[-which(names(test_data) == "id")], factor)

preds <- predict(log_wf, new_data=test_data, type="prob")
kaggle_submission <- preds %>%
  bind_cols(., test_data) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)  
  

## Write out the file
vroom_write(x=kaggle_submission, file="./LogRegPreds.csv", delim=",")


