# Coding club tutorial on logistic regression model 
# Author: Olivia Lin
# Contact: s1721244@sms.ed.ac.uk
# Last amended by Olivia Lin on 2 December 2019
----------------------------------------------------

# Please install these packages first if you don't already have them
# install.packages("DAAG")
# install.packages("dplyr")
# install.packages("caret")
# install.packages("ggplot2")
# install.packages("InformationValue")

# Load libraries ----

library(DAAG)  # for the "frogs" dataset
library(dplyr)  # for data manipulation
library(caret)  # for data splitting
library(ggplot2)  # for plotting graph
library(InformationValue)  # for finding optimal prob cutoff


# Explore the data ----

glimpse(frogs)


# 1. Prepare and clean the data ----

# Check if there is any missing value in each column
sapply(frogs, function(x)sum(is.na(x)))  # Summing the count of NA values in each column

lapply(frogs, function(x)sum(is.na(x)))  
# Here you can explore lappy(), which does the same thing as sapply 
# but returns output as a list

# No missing values are found, so we can proceed to the next step 

# Select only the columns we need
frogs <- select(frogs, -c(northing, easting))  # select all except northing and easting

# Change frogs' presence to factor variables
frogs$pres.abs <- as.factor(frogs$pres.abs)

# pres.abs is now factor variable presented as either 1 or 0


# 2. Data splitting ----

# Prepare Training and test data

# Set the seed to give us reproducible results
set.seed(100)  # 100 is the starting point of the generation of a sequence of random numbers

# Choose the rows for training data
train_data_index <- 
  createDataPartition(frogs$pres.abs, 
                      p = 0.7,  # 70% of the rows go to training 
                      list = F)  # return the results as a matrix but not a list

# Partitioning the data into two sets
train_data <- frogs[train_data_index, ]  # the chosen 70% of the rows as training data
test_data <- frogs[-train_data_index, ]  # choose the remaining rows as test data

# 3. Class imbalance ----

# Check the proportion of "0"s and "1"s in the training data
table(train_data$pres.abs)

# Find the proportion of "0"s and "1"s
table(frogs$pres.abs)

# the absence and presence response are split into approximately 1.7:1 ratio

# Upsampling ----

set.seed(100)  # Set the seed at 100 again

# Select the predictor variables columns
x_col <- select(train_data, -c(pres.abs))  # select everthing except "pres.abs" column

# Upsize the training sample 
up_train <- upSample(x = x_col,  # the predictor variables
                     y = train_data$pres.abs)  # the factor variable with class

# Check the proportion of "0"s and "1"s in the upsized training data
table(up_train$Class)

# the "1"s are upsized to 94, equal ratio achieved


# 4. The logistic regression model ----

# *A. Let's visualise our data first ----
(plot_1 <- 
    ggplot(up_train, 
           aes(x = altitude + distance + NoOfPools + NoOfSites + 
                   avrain + meanmin + meanmax, 
               y = as.numeric(as.character(Class)))) +  # change "Class" to numeric values
   geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +  # plot the classes of each observation
   geom_smooth(method = "glm",  # add the curve
               method.args = list(family = "binomial"),
               colour = "paleturquoise3",  # colour of the curve
               fill = "azure3",
               size = 1) +  # colour of the SE area
   theme_bw() +
   theme(panel.grid.minor = element_blank(),
         panel.grid.major.x= element_blank(),
         plot.margin = unit(c(1,1,1,1), units = , "cm"),
         plot.title = element_text(face = "bold", size = 10, hjust = 0),
         axis.text = element_text(size = 8),
         axis.title = element_text(size = 8)) +
   labs(title = "Logistic Regression Model 1\n",  #  "\n" indicates where the space is added
        y = "Probability of the presence of frogs\n",
        x = "\naltitude + distance + NoOfPools + NoOfSites+
        \navrain + meanmin + meanmax") +
  ggsave("image/plot_1.png", width = 5, height = 4, dpi = 800))

# Using generalised linear model
model <- glm(Class ~. ,  # "." indicated all other variables
             family = "binomial", data = up_train)

# *B. Checking assumptions ----

# Assumption 2: Multicollinearity ----
vif(model)

# The vif values should be < 5

# Adjust our model ----
model_new <- glm(Class ~ distance + NoOfPools + NoOfSites + avrain, 
                     family = "binomial", data = up_train)

# Compute the VIF values to check for multicollinearity again
vif(model_new)

# Assumption 3: Linearity ----
plot(model_new, which = 1)  # Call for the 1st plot

# Assumption 4: Independence ----
plot(model$residuals, type = "o")

# *C. Interpreting the results ----

summary(model_new)
anova(model_new, test = "Chisq")

# Comparing the graph of distance only and the previous plot_1
(plot_2 <- 
    ggplot(up_train, 
           aes(x = distance, 
               y = as.numeric(as.character(Class)))) +  
    geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +  
    geom_smooth(method = "glm",  # add the curve
                method.args = list(family = "binomial"),
                colour = "paleturquoise3", 
                fill = "azure3",
                size = 1) +
    theme_bw() +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x= element_blank(),
          plot.margin = unit(c(1,1,1,1), units = , "cm"),
          plot.title = element_text(face = "bold", size = 10, hjust = 0),
          axis.text = element_text(size = 8),
          axis.title = element_text(size = 8)) +
    labs(title = "Logistic Regression Model 2\n",  
         y = "Probability of the presence of frogs\n",
         x = "\nDistance to the nearest extant population (m)") +
    ggsave("image/plot_2.png", width = 5, height = 4, dpi = 800))


# *D. Predict the probabilities of presence on test data ----

# Add a new column for the predicted probabilities
test_data <- test_data %>%
  mutate(prob = plogis(predict(model_new, newdata = test_data)))

# alternatively, adding the response argument
test_data <- test_data %>%
  mutate(prob_2 = predict(model_new, newdata = test_data, 
                        type = "response"))  

# they gave the same results


# Decide on optimal prediction probability cutoff

# The default cutoff prediction probability score is 0.5
# To find the optimal probability cutoff 
opt_cut_off <- optimalCutoff(test_data$pres.abs, test_data$prob)

# the p cut-off that gives the minimum misclassification error = 0.519


# Categorise individuals into 2 classes based on their predicted probabilities
# So if probability of Y > 0.519, it will be classified as an event (Present = 1)

# Add a new column for the predicted class
test_data <- test_data %>% 
  mutate(predict_class = ifelse(prob > 0.519, 1, 0))

# Plot the prediction graph ----

(pred_plot <- ggplot(test_data, aes(x = distance, y = predict_class)) +
    geom_point(alpha = 0.2, colour = "rosybrown2", size = 2) +
    stat_smooth(method = "glm", 
                method.args = list(family = "binomial"),
                colour = "indianred",
                fill = "azure3",
                size = 1) +
    theme_bw() +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x= element_blank(),
          plot.margin = unit(c(1,1,1,1), units = , "cm"),
          plot.title = element_text(face = "bold", size = 10, hjust = 0),
          axis.text = element_text(size = 8),
          axis.title = element_text(size = 8)) +
    scale_y_continuous(limits = c(0,1)) +  # Set min and max y values at 0 and 1 respectively
    scale_x_continuous(limits = c(min(test_data$distance), max(test_data$distance))) +
    labs(title = "Predicting frogs' presence on test data\n",
         x = "\nDistance to nearest extant population (m)",  #  "\n" adds space above x-axis title
         y = "Probability of the presence of frogs\n") +
    ggsave("image/pred_plot.png", width = 5, height = 4, dpi = 800))

# *E. Model accuracy ----

# Check the data types again
glimpse(test_data)

# Change the pres.abs. to numeric values
test_data$pres.abs <- as.numeric(as.character(test_data$pres.abs))

# How much predicted presence match with the actual data in test data 
accuracy <- mean(test_data$predict_class == test_data$pres.abs)

# accuracy of the model = 72.6%


