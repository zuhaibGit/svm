library(numDeriv)

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

#Given a number, gives ~90% of them a lable of 1 and 10% of them a 
#label of 0. The number of rows in a data frame will be passed here
#to split them into test-train groups.
test_train_split <- function(num_rows) {
  newCol <- rbinom(n = num_rows, size = 1, prob = 0.9)
  split <- list(which(newCol == 1), which(newCol == 0))
  return(split)
}


do_step <- function(rows, lambda, a, b, batch_size, step_size) {
  batch <- sample(rows, batch_size, replace = F)
  
  #Finds the vector obtained from summation of gradients of batch
  batch_sum <- rep(0, ncol(training_data) + 1)
  for (i in batch) {
    gamma <- labels[i] * (sum(a * unlist(training_data[i,])) + b)
    #print(gamma)
    #Only adds to the summation if gamma < 1, becauses otherwise the gradient of the
    #funciton is the zero vector.
    if (gamma < 1) {
      batch_sum <- batch_sum + labels[i]*c(unlist(training_data[i,]),1)
    }
  }
  
  #print(batch_sum)
  
  direction <- (-1/length(batch))*batch_sum + lambda * c(a,0) 
  
  #print(direction)
  
  new_u = c(a,b) + step_size*direction
  return(new_u)
}

do_season <- function(rows, lambda, a, b, batch_size, step_size) {
  #Get held out set for the season
  hold_out <- sample(rows, 50, replace = F)
  #Vector to store accuracies agains hold out set, every 30 steps.
  store_accuracies <- c()
  
  #Take 300 steps. Every 30 steps, calculate accuracy of held out set.
  for (step in 1:300) {
    if ((step > 0) && ((step %% 30) == 0)) {
      #Test agains hold out set. Store the accuracy in a vector.
      predictions <- lapply(hold_out, function(x) {return(ifelse(sum(a*training_data[x,])+b > 0, 1, -1))})
      store_accuracies <- c(store_accuracies, length(which(predictions == labels[hold_out]))/50)
    }
    
    #Get updated values after stepping.
    new_u <- do_step(rows, lambda, a, b, batch_size, step_size)
    a <-  new_u[1:6]
    b <- new_u[7]
  }
  #print(paste("accurcies:", length(store_accuracies)))
  #print(length(c(store_accuracies,a,b)))
  return(c(store_accuracies, a, b))
}


batch_size <- 1
#Split the data into 90% training, and 10% validation data
split <- test_train_split(nrow(training_data))

held_out_accuracies_lambda_1 <- c()
a <- c(0,0,0,0,0,0)
b <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01 * k + 50)
  ret_vec <- do_season(split[[1]], 1e-4, a, b, batch_size, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a <- ret_vec[11:16]
  b <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_1 <- c(held_out_accuracies_lambda_1, ret_vec[1:10])
  print(paste("Done season", k))
}

held_out_accuracies_lambda_2 <- c()
a <- c(0, 0, 0, 0, 0, 0)
b <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01 * k + 50)
  ret_vec <- do_season(split[[1]], 1e-3, a, b, batch_size, step_size)
  a <- ret_vec[11:16]
  b <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_2 <- c(held_out_accuracies_lambda_2, ret_vec[1:10])
  print(paste("Done season", k))
}

held_out_accuracies_lambda_3 <- c()
a <- c(0, 0, 0, 0, 0, 0)
b <- 0
for (k in 1:100) {
  #print(paste("Lengths of a and b:", length(a), length(b)))
  step_size <- 1/(0.01 * k + 50)
  ret_vec <- do_season(split[[1]], 1e-2, a, b, batch_size, step_size)
  #print(paste("ret_vec:",length(ret_vec)))
  a <- ret_vec[11:16]
  b <- ret_vec[17]
  #print(c(a,b))
  held_out_accuracies_lambda_3 <- c(held_out_accuracies_lambda_3, ret_vec[1:10])
  print(paste("Done season", k))
}

held_out_accuracies_lambda_4 <- c()
a <- c(0, 0, 0, 0, 0, 0)
b <- 0
for (k in 1:100) {
  step_size <- 1/(0.01 * k + 50)
  ret_vec <- do_season(split[[1]], 1e-1, a, b, batch_size, step_size)
  a <- ret_vec[11:length(ret_vec) - 1]
  b <- ret_vec[length(ret_vec)]
  held_out_accuracies_lambda_1 <- c(held_out_accuracies_lambda_1, ret_vec[1:10])
}
