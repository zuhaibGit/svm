library(ggplot2)
library(dplyr)
library(tidyr)

################DATA SETUP#####################
setwd("C:/Users/Zuhaib Ahmed/Desktop/Machine Learning/Assignment 2")

training_data <- read.csv("training_data.csv")
training_data <- training_data[,c(1,3,5,11,12,13,15)]
testing_data <- read.csv("testing_data.csv")
testing_data <- testing_data[,c(1,3,5,11,12,13)]

#Scales the feature variables to have mean of 0 and variance of 1
training_means <- apply(training_data[,1:6], 2, mean)
training_sds <- apply(training_data[,1:6], 2, sd)

testing_means <- apply(testing_data, 2, mean)
testing_sds <- apply(testing_data, 2, sd)

#training_data[,1:6] <- ((training_data[,1:6] - training_means)/training_vars)
training_data[,1:6] <- as.data.frame(Map("-", training_data[,1:6], training_means))
training_data[,1:6] <- as.data.frame(Map("/", training_data[,1:6], training_sds))
#testing_data <- apply(testing_data, 2, function(x) {return((x - mean(x))/var(x))})
testing_data <- as.data.frame(Map("-", testing_data, testing_means))
testing_data <- as.data.frame(Map("/", testing_data, testing_sds))
names(training_data) <- c("age", "fnlwgt", "edu-num", "gain", "loss", "hours", "income")
income <- training_data[,7]
labels <- ifelse(income == income[1], 1, -1)
training_data <- training_data[,1:6]
names(testing_data) <- c("age", "fnlwgt", "edu-num", "gain", "loss", "hours")


######################FUNCTIONS#########################
#Given a number, gives ~90% of them a lable of 1 and 10% of them a 
#label of 0. The number of rows in a data frame will be passed here
#to split them into test-train groups.
test_train_split <- function(num_rows) {
  newCol <- rbinom(n = num_rows, size = 1, prob = 0.9)
  split <- list(which(newCol == 1), which(newCol == 0))
  return(split)
}

#Performs a step of the stochastic gradient descent process. Returns a vector containing
#a and b.
do_step <- function(rows, lambda, a, b, batch_size, step_size) {
  batch <- sample(rows, 1)
  gamma <- labels[batch]*(sum(a*unlist(training_data[batch,])) + b)
  if (gamma >= 1) {
    a <- a - step_size*lambda*a
  } else {
    a <- a - step_size*(lambda*a - labels[batch]*unlist(training_data[batch,]))
    b <- b - step_size*(-labels[batch])
  }
  return(c(a,b))
}

#Does an iteration of a season. Involves taking 300 steps. Every 30 steps, it checks
#the current a and b values agains the hold-out set, and keeps track of the accuracy.
#It also keeps track of the magnitude of the vector a. The function returns 
#the accuracies (every 30 steps), a, b, and the magnitude of a (every 30 steps)
do_season <- function(rows, lambda, a, b, batch_size, step_size) {
  #Get held out set for the season
  hold_out <- sample(rows, 50, replace = F)
  
  #Vectors to store accuracies agains hold out set and magnitude of a, every 30 steps.
  store_accuracies <- c()
  mag_a <- c()
  
  #Take 300 steps. Every 30 steps, calculate accuracy of held out set.
  for (step in 1:300) {
    if ((step > 0) && ((step %% 30) == 0)) {
      #Test agains hold out set. Store the accuracy in a vector.
      predictions <- lapply(hold_out, function(x) {return(ifelse(sum(a*training_data[x,])+b > 0, 1, -1))})
      store_accuracies <- c(store_accuracies, length(which(predictions == labels[hold_out]))/50)
      mag_a <- c(mag_a, sum(a*a))
    }
    
    #Get updated values after stepping.
    new_u <- do_step(rows, lambda, a, b, batch_size, step_size)
    a <-  new_u[1:6]
    b <- new_u[7]
  }
  return(c(store_accuracies, a, b, mag_a))
}

########LEARNING AND CROSS VALIDATION##############
batch_size <- 1
#Split the data into 90% training, and 10% validation data
split <- test_train_split(nrow(training_data))

held_out_accuracies_lambda_1 <- c()
coefficient_magnitude_1 <- c()
a_1 <- c(0,0,0,0,0,0)
b_1 <- 1
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01*k + 50)
  ret_vec <- do_season(split[[1]], 1e-4, a_1, b_1, 1, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a_1 <- ret_vec[11:16]
  b_1 <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_1 <- c(held_out_accuracies_lambda_1, ret_vec[1:10])
  coefficient_magnitude_1 <- c(coefficient_magnitude_1, ret_vec[18:27])
  
  print(paste("Done season", k))
}
test_predictions_1 <- lapply(split[[2]], function(x) {return(ifelse(sum(a_1*training_data[x,])+b_1> 0, 1, -1))})
accuracy_1 <- length(which(test_predictions_1 == labels[split[[2]]]))/length(split[[2]])


held_out_accuracies_lambda_2 <- c()
coefficient_magnitude_2 <- c()
a_2 <- c(0,0,0,0,0,0)
b_2 <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01*k + 50)
  ret_vec <- do_season(split[[1]], 1e-3, a_2, b_2, 1, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a_2 <- ret_vec[11:16]
  b_2 <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_2 <- c(held_out_accuracies_lambda_2, ret_vec[1:10])
  coefficient_magnitude_2 <- c(coefficient_magnitude_2, ret_vec[18:27])
  
  print(paste("Done season", k))
}
test_predictions_2 <- lapply(split[[2]], function(x) {return(ifelse(sum(a_2*training_data[x,])+b_2 > 0, 1, -1))})
accuracy_2 <- length(which(test_predictions_2 == labels[split[[2]]]))/length(split[[2]])


held_out_accuracies_lambda_3 <- c()
coefficient_magnitude_3 <- c()
a_3 <- c(0,0,0,0,0,0)
b_3 <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01*k + 50)
  ret_vec <- do_season(split[[1]], 1e-2, a_3, b_3, 1, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a_3 <- ret_vec[11:16]
  b_3 <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_3 <- c(held_out_accuracies_lambda_3, ret_vec[1:10])
  coefficient_magnitude_3 <- c(coefficient_magnitude_3, ret_vec[18:27])
  
  print(paste("Done season", k))
}
test_predictions_3 <- lapply(split[[2]], function(x) {return(ifelse(sum(a_3*training_data[x,])+b_3 > 0, 1, -1))})
accuracy_3 <- length(which(test_predictions_3 == labels[split[[2]]]))/length(split[[2]])


held_out_accuracies_lambda_4 <- c()
coefficient_magnitude_4 <- c()
a_4 <- c(0,0,0,0,0,0)
b_4 <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01*k + 50)
  ret_vec <- do_season(split[[1]], 1e-1, a_4, b_4, 1, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a_4 <- ret_vec[11:16]
  b_4 <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_4 <- c(held_out_accuracies_lambda_4, ret_vec[1:10])
  coefficient_magnitude_4 <- c(coefficient_magnitude_4, ret_vec[18:27])
  
  print(paste("Done season", k))
}
test_predictions_4 <- lapply(split[[2]], function(x) {return(ifelse(sum(a_4*training_data[x,])+b_4 > 0, 1, -1))})
accuracy_4 <- length(which(test_predictions_4 == labels[split[[2]]]))/length(split[[2]])

###########################PLOT GENERATION###########################33
df_coefficient_magnitudes <- data.frame(coefficient_magnitude_1, coefficient_magnitude_2,
                                        coefficient_magnitude_3, coefficient_magnitude_4)
names(df_coefficient_magnitudes) <- c("1e-4", "1e-3", "1e-2", "1e-1")
df_coefficient_magnitudes %>% 
  gather(Var, Val) %>% 
  mutate(x = rep(1:1000, 4)) %>% 
  ggplot(aes(x, Val)) + 
  geom_line(aes(color = Var)) + 
  xlab("Steps (incremented by 30)") +
  ylab("Magnitude of Coefficient") +
  ggtitle("Coefficient Magnitude Variation for \nMultiple Regularization Constants") +
  scale_color_manual(values = c("black", "blue", "green", "yellow"))

  
df_accuracies <- data.frame(held_out_accuracies_lambda_1, held_out_accuracies_lambda_2,
                            held_out_accuracies_lambda_3, held_out_accuracies_lambda_4)
names(df_accuracies) <- c("1e-4", "1e-3", "1e-2", "1e-1")
df_accuracies %>% 
  gather(Var, Val) %>% 
  mutate(x = rep(1:1000, 4)) %>% 
  ggplot(aes(x, Val)) + 
  geom_line(aes(color = Var)) + 
  xlab("Steps (incremented by 30)") +
  ylab("Held out accuracy") +
  ggtitle("Held Out Accuracies with \nMultiple Regularization Constants") +
  scale_color_manual(values = c("light green", "black", "light blue", "light yellow"))

  

#################FINAL TEST FOR SUBMISSION######################
#I trained on the entire data set, and did more seasons so that I could be more confident
#that we saw all the points, and of convergence.
a_final <- c(0,0,0,0,0,0)
b_final <- 0
for (k in 1:450) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01*k + 50)
  ret_vec <- do_season(c(1:nrow(training_data)), 1e-2, a_final, b_final, 1, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a_final <- ret_vec[11:16]
  b_final <- ret_vec[17]
  print(paste("Done season", k))
}
test_predictions_final <- lapply(c(1:nrow(testing_data)), function(x) {return(ifelse(sum(a_final*testing_data[x,])+b_final > 0, 1, -1))})
test_predictions_final_labels <- lapply(test_predictions_final, function(x) {return(ifelse(x == 1, ">50K", "<=50K"))})



sink("submission.txt")
rapply(test_predictions_final_labels, print)
sink()