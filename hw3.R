library(ggplot2)

#each column or each vector is one training example
#each row are classifiers

test = read.csv('mnist_test.csv', header = FALSE)
train = read.csv('mnist_train.csv',header = FALSE)

test_0_1 = test[, (test[nrow(test),] == 0 | test[nrow(test),] == 1)]
true_label_test_0_1 = test_0_1[nrow(test_0_1),]
test_0_1 = test_0_1[1:nrow(test_0_1)-1,]

test_3_5 = test[, (test[nrow(test),] == 3 | test[nrow(test),] == 5)]
true_label_test_3_5 = test_3_5[nrow(test_3_5),]
test_3_5 = test_3_5[1:nrow(test_3_5)-1,]

train_0_1 = train[, (train[nrow(test),] == 0 | train[nrow(test),] == 1)]
true_label_train_0_1 = train_0_1[nrow(train_0_1),]
train_0_1 = train_0_1[1:nrow(train_0_1)-1,]

train_3_5 = train[, (train[nrow(test),] == 3 | train[nrow(test),] == 5)]
true_label_train_3_5 = train_3_5[nrow(train_3_5),]
train_3_5 = train_3_5[1:nrow(train_3_5)-1,]

rotate = function(x) t(apply(x, 2, rev))

image_0 = as.matrix(test_0_1[,1])
image(rotate(matrix(image_0,28,28)),col=gray.colors(256))

image_1 = as.matrix(test_0_1[,1500])
image(rotate(matrix(image_1,28,28)),col=gray.colors(256))

image_3 = as.matrix(test_3_5[,1])
image(rotate(matrix(image_3,28,28)),col=gray.colors(256))

image_5 = as.matrix(test_3_5[,1900])
image(rotate(matrix(image_5,28,28)),col=gray.colors(256))

############### Question 2

#implementation of logistic regression
gradient = function(x, y, theta) {
  y = as.matrix(y)
  grad = (1 / nrow(y)) * (t(x) %*% (1/(1 + exp(-x %*% t(theta))) - y))
  return (t(grad))
}

# Gradient descent function
gradient_descent <- function(theta,x, y, alpha=0.1, iterations=500, threshold=1e-5) {
  
  # Add x_0 = 1 as the first column
  m = nrow(x)
  x = cbind(rep(1,m),x)
  
  # Look at the values over each iteration
  stack_theta = theta
  for (i in 1:iterations) {
    theta = theta - alpha * gradient(x, y, theta)
    if(all(is.na(theta))) break
    stack_theta = rbind(stack_theta, theta)
    
    if(i > 2){
      if(all(abs(theta - stack_theta[i-1,]) < threshold)) break 
    } 
  }
  return (theta)
}

#Sigmoid function
sigmoid = function(z){
  g = 1/(1+exp(-z))
  return (g)
}

calculate_probability = function(theta, x, y){
  g = sigmoid(y*(x%*%theta))
  return (g)
}

probability = function(n,x,y,theta){
  prob = calculate_probability(theta, x[n,], y[n,])
  return (prob)
}

score = function(x, y, theta){
  
  # Add x_0 = 1 as the first column
  m = nrow(x)
  x = cbind(rep(1,m),x)
  
  count_0 = 0
  count_1 = 0
  count_3 = 0
  count_5 = 0
  count = 0
  
  prob = (sapply(1:nrow(x), probability, x, y, theta))
  prob = as.matrix(prob)
  
  for(n in 1:nrow(x)){
    #when the label is 0 and probability is less than equal to 0.5
    if(y[n, ]==0 && prob[n, ] <= 0.5){
      count_0 = count_0 + 1
    }
    #when the label is 0 and probability is greater than 0.5
    else if(y[n,]==1 && prob[n, ] > 0.5){
      count_1 = count_1 + 1
    }
    else if(y[n,]==3 && prob[n, ] <= 0.5){
      count_3 = count_3 + 1
    }
    #when the label is 0 and probability is less than equal to 0.5
    else if(y[n,]==5 && prob[n, ] > 0.5){
      count_5 = count_5 + 1
    }
    #when none of the above four cases are fulfilled
    else{
      count = count + 1
    }
  }
  accuracy = ((nrow(x)-count)/nrow(x))*100
  return (accuracy)
}

initialize_theta = function(x){
  # Initialize the vector theta
  initial_theta = matrix(rep(0, (ncol(x)+1)), nrow=1)
  return (initial_theta)
}

############### Question 3

## 3.a

y_train_3_5 = true_label_train_3_5
y_train_3_5[y_train_3_5 == 3] = 0
y_train_3_5[y_train_3_5 == 5] = 1
y_train_3_5_transpose = t(y_train_3_5)

y_test_3_5 = true_label_test_3_5
y_test_3_5[y_test_3_5 == 3] = 0
y_test_3_5[y_test_3_5 == 5] = 1
y_test_3_5_transpose = t(y_test_3_5)

#number of iterations for convergence of theta
iterations = 300

train_0_1_transpose = t(train_0_1)
true_label_train_0_1_transpose = t(true_label_train_0_1)

train_3_5_transpose = t(train_3_5)
true_label_train_3_5_transpose = t(true_label_train_3_5)

test_0_1_transpose = t(test_0_1)
true_label_test_0_1_transpose = t(true_label_test_0_1)

test_3_5_transpose = t(test_3_5)
true_label_test_3_5_transpose = t(true_label_test_3_5)

#vector theta for traning set 0_1
theta_0_1 = gradient_descent(theta=initialize_theta(train_0_1_transpose), x=train_0_1_transpose, y=true_label_train_0_1_transpose, iterations=iterations)
theta_0_1 = as.matrix(theta_0_1)

#vector theta for traning set 3_5 with training set having labels 0 and 1
theta_3_5 = gradient_descent(theta=initialize_theta(train_3_5_transpose), x=train_3_5_transpose, y=y_train_3_5_transpose, iterations=iterations)
theta_3_5 = as.matrix(theta_3_5)

#accuracy on test 0_1 using vector theta for 0_1
accuracy_test_0_1 = score(x=test_0_1_transpose, y=true_label_test_0_1_transpose, theta=t(theta_0_1))
#accuracy_test_0_1 = 100

#accuracy on test 3_5 converted to 0 & 1 using vector theta for 3_5 
accuracy_test_3_5 = score(x=test_3_5_transpose, y=y_test_3_5_transpose,theta=t(theta_3_5))
#accuracy_test_3_5 = 97.58149

#accuracy on train 0_1 using vector theta for 0_1
accuracy_train_0_1 = score(x=train_0_1_transpose, y=true_label_train_0_1_transpose,theta=t(theta_0_1))
#accuracy_train_0_1 = 99.91315

#accuracy on train 3_5 converted to 0 & 1 using vector theta for 3_5
accuracy_train_3_5 = score(x=train_3_5_transpose, y=y_train_3_5_transpose,theta=t(theta_3_5))
#accuracy_train_3_5 = 97.73199

## 3.b

#general function which accepts train data and test data to find accuracy
accuracy_test_train = function(n,x,y,iterations,x1,y1){
  
  #random shuffling data row-wise:
  x = x[sample(nrow(x)),]

  #getting vector theta after traning the model
  theta_trained = gradient_descent(theta=initialize_theta(x), x=x, y=y, iterations=iterations)
  theta_trained = as.matrix(theta_trained)
  
  #getting accuracy on vector theta
  accuracy = score(x=x1, y=y1, theta=t(theta_trained))
  
  return (accuracy)
}

#obtaining 10 accuracies for test_0_1 by training the model 10 times
accuracy_test_0_1_10times = sapply(1:10,accuracy_test_train,x=train_0_1_transpose,y=true_label_train_0_1_transpose,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose)
#accuracy_test_0_1_10times = 46.33570 100.00000 100.00000 100.00000 100.00000 100.00000 100.00000  58.77069 100.00000  46.33570
#mean(accuracy_test_0_1_10times) = 85.14421

#obtaining 10 accuracies for test_3_5 by training the model 10 times
accuracy_test_3_5_10times = sapply(1:10,accuracy_test_train,x=train_3_5_transpose,y=y_train_3_5_transpose,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose)
#accuracy_test_3_5_10times = 85.48896 100.00000 100.00000  76.02524  53.10200  53.15457  53.10200  53.10200  53.10200 100.00000
#mean(accuracy_test_3_5_10times) = 76.88749

#obtaining 10 accuracies for train_0_1 by training the model 10 times
accuracy_train_0_1_10times = sapply(1:10,accuracy_test_train,x=train_0_1_transpose,y=true_label_train_0_1_transpose,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose)
#accuracy_train_0_1_10times = 99.92894 100.00000  46.76668 100.00000 100.00000  46.76668 100.00000  56.88117  46.76668  46.76668
#mean(accuracy_train_0_1_10times) = 74.38768

#obtaining 10 accuracies for train_3_5 by training the model 10 times
accuracy_train_3_5_10times = sapply(1:10,accuracy_test_train,x=train_3_5_transpose,y=y_train_3_5_transpose,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose)
#accuracy_train_3_5_10times = 53.07306  90.39127  53.07306  53.07306 100.00000  53.07306  98.88331  53.07306  99.65374  53.07306
#mean(accuracy_train_3_5_10times) = 70.73667

############### Question 4

## 4.a

#this function will return different initializations of the parameter (intial vector theta)
initialize_param = function(x,theta_value){
  
  if(theta_value==1){
  #Initialize the vector theta with (n+1) random values between 0 & 1 where n is the number of features
    initial_theta = matrix(runif(ncol(x)+1),nrow=1)
  }
  else{
  # Initialize the vector theta with theta_value
  initial_theta = matrix(rep(theta_value, (ncol(x)+1)), nrow=1)
  }
  return (initial_theta)
}

#general function which accepts train data and test data and initial value of theta to find accuracy
#This function can be called n number of times to find n accuracies of test and traning for 
#classification 0_1 and 3_5
accuracy_with_different_InitialTheta = function(n,x,y,initial_theta,iterations,x1,y1){
  
  #random shuffling data row-wise:
  x = x[sample(nrow(x)),]
  
  #getting vector theta after traning the model
  theta_trained = gradient_descent(theta=initialize_param(x,initial_theta), x=x, y=y, iterations=iterations)
  theta_trained = as.matrix(theta_trained)
  
  #getting accuracy on vector theta
  accuracy = score(x=x1, y=y1, theta=t(theta_trained))
  
  return (accuracy)
}

#obtaining 10 accuracies for test_3_5 by training the model 10 times on different values of theta - 0, 0.5
accuracy_test_3_5_10times_InitialTheta_0 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose)
#accuracy_test_3_5_10times = 100.00000 100.00000  70.76761  53.10200  85.80442  53.10200 100.00000  99.89485  53.10200  53.10200
#mean(accuracy_test_3_5_10times) = 76.88749

accuracy_test_3_5_10times_InitialTheta_0.5 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.5,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose)
#accuracy_test_3_5_10times_InitialTheta_0.5 = 53.10200  53.10200  53.10200 100.00000  78.96951  90.64143  53.10200  53.20715  99.94742  90.79916
#mean(accuracy_test_3_5_10times_InitialTheta_0.5) = 72.59727

accuracy_test_3_5_10times_InitialTheta_0.9 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.9,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose)
#accuracy_test_3_5_10times_InitialTheta_0.9 = 99.15878  53.10200  97.16088  53.10200  53.10200 100.00000 100.00000  53.10200  53.10200 100.00000
#mean(accuracy_test_3_5_10times_InitialTheta_0.9) = 76.18297

#obtaining 10 accuracies for train_3_5 by training the model 10 times
accuracy_train_3_5_10times_InitialTheta_0 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose)
#accuracy_train_3_5_10times_InitialTheta_0 =  88.49550  53.07306  53.07306 100.00000  99.99134 100.00000  53.07306  89.80263  53.07306  53.07306
#mean(accuracy_train_3_5_10times_InitialTheta_0) = 74.36548

accuracy_train_3_5_10times_InitialTheta_0.5 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.5,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose)
#accuracy_train_3_5_10times_InitialTheta_0.5 = 99.99134  84.39231  99.54120  53.07306  53.07306  53.07306  53.07306 100.00000  53.07306  53.07306
#mean(accuracy_train_3_5_10times_InitialTheta_0.5) = 70.23632

accuracy_train_3_5_10times_InitialTheta_0.9 = sapply(1:10,accuracy_with_different_InitialTheta,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.9,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose)
#accuracy_train_3_5_10times_InitialTheta_0.9 = 96.40755 98.29467 53.07306 53.07306 53.07306 99.58449 53.07306 99.80956 53.07306 53.07306
#mean(accuracy_train_3_5_10times_InitialTheta_0.9) = 71.25346

## 4b

# new Gradient descent function containing convergence criteria
gradient_descent_convergence_criteria = function(theta,x, y, alpha=0.1, iterations=500, threshold=1e-5, convergence_criteria=1) {
  
  # Add x_0 = 1 as the first column
  m = nrow(x)
  x = cbind(rep(1,m),x)
  
  # Look at the values over each iteration
  stack_theta = theta
  for (i in 1:iterations) {
    theta = theta - alpha * gradient(x, y, theta)
    if(all(is.na(theta))) break
    stack_theta = rbind(stack_theta, theta)
  
    if(i > 2){
      if(convergence_criteria==1){
        ##convergence criteria for absolute convergence
        if(all(abs(theta - stack_theta[i-1,]) < threshold)) break 
      }
      else if(convergence_criteria==2){
        #convergence criteria for relative convergence
        if(all(abs((theta - stack_theta[i-1,])/stack_theta[i-1,]) < threshold)) break
      }
    } 
  }
  return (theta)
}

#new function which accepts train data and test data, initial value of theta and convergence criteria 
#to find accuracy. This function can be called n number of times to find n accuracies of test and 
#traning for classification 0_1 and 3_5
accuracy_with_different_convergence_criteria = function(n,x,y,initial_theta,iterations,x1,y1,convergence){
  
  #random shuffling data row-wise:
  x = x[sample(nrow(x)),]
  
  #getting vector theta after traning the model
  theta_trained = gradient_descent_convergence_criteria(theta=initialize_param(x,initial_theta), x=x, y=y,iterations=iterations,convergence_criteria=convergence)
  theta_trained = as.matrix(theta_trained)
  
  #getting accuracy on vector theta
  accuracy = score(x=x1, y=y1, theta=t(theta_trained))
  
  return (accuracy)
}

#obtaining 10 accuracies for test_3_5 by training the model 10 times on different values of theta - 0, 0.5, 0.9 and different convergence
accuracy_test_3_5_InitialTheta_0_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=2)
#accuracy_test_3_5_InitialTheta_0_convergence_2 = 65.66772 64.77392 66.61409 65.14196 66.14090 65.24711 68.45426 67.08728 65.66772 64.45846
#mean(accuracy_test_3_5_InitialTheta_0_convergence_2) = 65.92534

accuracy_test_3_5_InitialTheta_0.5_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.5,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=2)
#accuracy_test_3_5_InitialTheta_0.5_convergence_2 = 67.66562 69.45321 70.08412 67.82334 68.50683 68.55941 68.55941 66.98212 68.71714 67.45531
#mean(accuracy_test_3_5_InitialTheta_0.5_convergence_2) = 68.38065

accuracy_test_3_5_InitialTheta_0.9_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.9,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=2)
#accuracy_test_3_5_InitialTheta_0.9_convergence_2 = 72.18717 73.55415 72.76551 72.60778 72.13460 72.39748 72.66036 72.92324 73.18612 72.76551
#mean(accuracy_test_3_5_InitialTheta_0.9_convergence_2) = 72.71819

#obtaining 10 accuracies for train_3_5 by training the model 10 times on different values of theta - 0, 0.5, 0.9 and different convergence
accuracy_train_3_5_InitialTheta_0_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose,convergence=2)
#accuracy_train_3_5_InitialTheta_0_convergence_2 = 62.41343 62.50866 63.85907 61.29675 61.11496 63.81579 64.69010 63.40028 61.64301 60.07618
#mean(accuracy_train_3_5_InitialTheta_0_convergence_2) = 62.48182

accuracy_train_3_5_InitialTheta_0.5_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.5,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose,convergence=2)
#accuracy_train_3_5_InitialTheta_0.5_convergence_2 = 72.32514 71.17382 71.00935 71.14785 72.05679 71.66724 71.55471 72.70602 71.83172 71.15651
#mean(accuracy_train_3_5_InitialTheta_0.5_convergence_2) = 71.66291

accuracy_train_3_5_InitialTheta_0.9_convergence_2 = sapply(1:10,accuracy_with_different_convergence_criteria,x=train_3_5_transpose,y=y_train_3_5_transpose,initial_theta=0.9,iterations=iterations, x1=train_3_5_transpose, y1=y_train_3_5_transpose,convergence=2)
#accuracy_train_3_5_InitialTheta_0.9_convergence_2 = 73.92659 73.58899 73.86600 73.55436 74.28151 73.97853 73.40720 74.10838 74.01316 74.08241
#mean(accuracy_train_3_5_InitialTheta_0.9_convergence_2) = 73.88071

############### Question 5

## 5a

percentage_subset = function(percentage, x){
  x_percentage = x[1:floor((percentage*nrow(x))/100), ]
  return (x_percentage)
}

accuracy_test_0_1_InitialTheta_0_convergence_1_set5 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(5,train_0_1_transpose),y=percentage_subset(5,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set5) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set10 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(10,train_0_1_transpose),y=percentage_subset(10,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set10) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set15 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(15,train_0_1_transpose),y=percentage_subset(15,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set15) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set20 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(20,train_0_1_transpose),y=percentage_subset(20,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set20) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set25 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(25,train_0_1_transpose),y=percentage_subset(25,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set25) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set30 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(30,train_0_1_transpose),y=percentage_subset(30,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set30) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set35 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(35,train_0_1_transpose),y=percentage_subset(35,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set35) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set40 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(40,train_0_1_transpose),y=percentage_subset(40,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set40) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set45 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(45,train_0_1_transpose),y=percentage_subset(45,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set45) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set50 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(50,train_0_1_transpose),y=percentage_subset(50,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set50) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set55 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(55,train_0_1_transpose),y=percentage_subset(55,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set55) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set60 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(60,train_0_1_transpose),y=percentage_subset(60,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set60) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set65 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(65,train_0_1_transpose),y=percentage_subset(65,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set65) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set70 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(70,train_0_1_transpose),y=percentage_subset(70,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set70) = 46.3357
accuracy_test_0_1_InitialTheta_0_convergence_1_set75 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(75,train_0_1_transpose),y=percentage_subset(75,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set75) = 46.35461
accuracy_test_0_1_InitialTheta_0_convergence_1_set80 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(80,train_0_1_transpose),y=percentage_subset(80,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set80) = 46.41135
accuracy_test_0_1_InitialTheta_0_convergence_1_set85 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(85,train_0_1_transpose),y=percentage_subset(85,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set85) = 48.01891
accuracy_test_0_1_InitialTheta_0_convergence_1_set90 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(90,train_0_1_transpose),y=percentage_subset(90,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set90) = 58.80851
accuracy_test_0_1_InitialTheta_0_convergence_1_set95 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(95,train_0_1_transpose),y=percentage_subset(95,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set95) = 80.89835
accuracy_test_0_1_InitialTheta_0_convergence_1_set100 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(100,train_0_1_transpose),y=percentage_subset(100,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=test_0_1_transpose, y1=true_label_test_0_1_transpose,convergence=1)
# mean(accuracy_test_0_1_InitialTheta_0_convergence_1_set100) = 95.3948

accuracy_test_0_1_set = c("46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.3357","46.35461","46.41135","48.01891","58.80851","80.89835","95.3948")

accuracy_test_3_5_InitialTheta_0_convergence_1_set5 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(5,train_3_5_transpose),y=percentage_subset(5,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set5) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set10 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(10,train_3_5_transpose),y=percentage_subset(10,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set10) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set15 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(15,train_3_5_transpose),y=percentage_subset(15,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set15) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set20 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(20,train_3_5_transpose),y=percentage_subset(20,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set20) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set25 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(25,train_3_5_transpose),y=percentage_subset(25,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set25) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set30 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(30,train_3_5_transpose),y=percentage_subset(30,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set30) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set35 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(35,train_3_5_transpose),y=percentage_subset(35,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set35) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set40 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(40,train_3_5_transpose),y=percentage_subset(40,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set40) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set45 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(45,train_3_5_transpose),y=percentage_subset(45,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set45) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set50 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(50,train_3_5_transpose),y=percentage_subset(50,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set50) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set55 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(55,train_3_5_transpose),y=percentage_subset(55,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set55) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set60 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(60,train_3_5_transpose),y=percentage_subset(60,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set60) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set65 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(65,train_3_5_transpose),y=percentage_subset(65,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set65) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set70 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(70,train_3_5_transpose),y=percentage_subset(70,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set70) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set75 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(75,train_3_5_transpose),y=percentage_subset(75,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set75) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set80 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(80,train_3_5_transpose),y=percentage_subset(80,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set80) = 53.102
accuracy_test_3_5_InitialTheta_0_convergence_1_set85 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(85,train_3_5_transpose),y=percentage_subset(85,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set85) = 53.1756
accuracy_test_3_5_InitialTheta_0_convergence_1_set90 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(90,train_3_5_transpose),y=percentage_subset(90,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set90) = 54.11146
accuracy_test_3_5_InitialTheta_0_convergence_1_set95 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(95,train_3_5_transpose),y=percentage_subset(95,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set95) = 55.37329
accuracy_test_3_5_InitialTheta_0_convergence_1_set100 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(100,train_3_5_transpose),y=percentage_subset(100,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set100) = 61.44059

accuracy_test_3_5_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.1756","54.11146","55.37329","61.44059")

accuracy_train_0_1_InitialTheta_0_convergence_1_set5 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(5,train_0_1_transpose),y=percentage_subset(5,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set5) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set10 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(10,train_0_1_transpose),y=percentage_subset(10,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set10) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set15 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(15,train_0_1_transpose),y=percentage_subset(15,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set15) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set20 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(20,train_0_1_transpose),y=percentage_subset(20,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set20) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set25 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(25,train_0_1_transpose),y=percentage_subset(25,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set25) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set30 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(30,train_0_1_transpose),y=percentage_subset(30,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set30) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set35 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(35,train_0_1_transpose),y=percentage_subset(35,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set35) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set40 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(40,train_0_1_transpose),y=percentage_subset(40,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set40) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set45 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(45,train_0_1_transpose),y=percentage_subset(45,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set45) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set50 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(50,train_0_1_transpose),y=percentage_subset(50,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set50) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set55 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(55,train_0_1_transpose),y=percentage_subset(55,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set55) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set60 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(60,train_0_1_transpose),y=percentage_subset(60,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set60) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set65 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(65,train_0_1_transpose),y=percentage_subset(65,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set65) = 46.76668
accuracy_train_0_1_InitialTheta_0_convergence_1_set70 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(70,train_0_1_transpose),y=percentage_subset(70,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set70) = 46.76984
accuracy_train_0_1_InitialTheta_0_convergence_1_set75 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(75,train_0_1_transpose),y=percentage_subset(75,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set75) = 46.78089
accuracy_train_0_1_InitialTheta_0_convergence_1_set80 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(80,train_0_1_transpose),y=percentage_subset(80,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set80) = 46.83616
accuracy_train_0_1_InitialTheta_0_convergence_1_set85 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(85,train_0_1_transpose),y=percentage_subset(85,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set85) = 47.42992
accuracy_train_0_1_InitialTheta_0_convergence_1_set90 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(90,train_0_1_transpose),y=percentage_subset(90,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set90) =  57.59811
accuracy_train_0_1_InitialTheta_0_convergence_1_set95 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(95,train_0_1_transpose),y=percentage_subset(95,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set95) = 82.19976
accuracy_train_0_1_InitialTheta_0_convergence_1_set100 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(100,train_0_1_transpose),y=percentage_subset(100,true_label_train_0_1_transpose),initial_theta=0,iterations=iterations, x1=train_0_1_transpose, y1=true_label_train_0_1_transpose,convergence=1)
# mean(accuracy_train_0_1_InitialTheta_0_convergence_1_set100) = 97.61548

accuracy_train_0_1_set = c("46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76668","46.76984","46.78089","46.83616","47.42992","57.59811","82.19976","97.61548")

accuracy_train_3_5_InitialTheta_0_convergence_1_set5 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(5,train_3_5_transpose),y=percentage_subset(5,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set5) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set10 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(10,train_3_5_transpose),y=percentage_subset(10,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set10) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set15 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(15,train_3_5_transpose),y=percentage_subset(15,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set15) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set20 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(20,train_3_5_transpose),y=percentage_subset(20,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set20) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set25 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(25,train_3_5_transpose),y=percentage_subset(25,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set25) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set30 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(30,train_3_5_transpose),y=percentage_subset(30,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set30) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set35 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(35,train_3_5_transpose),y=percentage_subset(35,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set35) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set40 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(40,train_3_5_transpose),y=percentage_subset(40,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set40) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set45 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(45,train_3_5_transpose),y=percentage_subset(45,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set45) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set50 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(50,train_3_5_transpose),y=percentage_subset(50,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set50) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set55 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(55,train_3_5_transpose),y=percentage_subset(55,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set55) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set60 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(60,train_3_5_transpose),y=percentage_subset(60,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set60) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set65 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(65,train_3_5_transpose),y=percentage_subset(65,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set65) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set70 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(70,train_3_5_transpose),y=percentage_subset(70,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set70) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set75 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(75,train_3_5_transpose),y=percentage_subset(75,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set75) = 53.102
accuracy_train_3_5_InitialTheta_0_convergence_1_set80 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(80,train_3_5_transpose),y=percentage_subset(80,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set80) = 53.12303
accuracy_train_3_5_InitialTheta_0_convergence_1_set85 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(85,train_3_5_transpose),y=percentage_subset(85,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set85) = 53.2387
accuracy_train_3_5_InitialTheta_0_convergence_1_set90 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(90,train_3_5_transpose),y=percentage_subset(90,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set90) =  53.73291
accuracy_test_3_5_InitialTheta_0_convergence_1_set95 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(95,train_3_5_transpose),y=percentage_subset(95,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_test_3_5_InitialTheta_0_convergence_1_set95) = 55.63617
accuracy_train_3_5_InitialTheta_0_convergence_1_set100 = sapply(1:5,accuracy_with_different_convergence_criteria,x=percentage_subset(100,train_3_5_transpose),y=percentage_subset(100,y_train_3_5_transpose),initial_theta=0,iterations=iterations, x1=test_3_5_transpose, y1=y_test_3_5_transpose,convergence=1)
# mean(accuracy_train_3_5_InitialTheta_0_convergence_1_set100) = 61.98738

accuracy_train_3_5_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.12303","53.2387","53.73291","55.63617","61.98738")

qplot(x=accuracy_test_3_5_set, y=accuracy_train_3_5_set,geom="point",xlab="accuracy of test",ylab ="accuracy of train", main="test vs train for classification 3_5")
qplot(x=accuracy_test_0_1_set, y=accuracy_train_0_1_set,geom="point",xlab="accuracy of test",ylab ="accuracy of train", main="test vs train for classification 0_1")

## 5b

log_likelihood = function(n,x,y,theta,iterations,convergence_criteria,x1,y1){
    
    # Add x_0 = 1 as the first column
    m = nrow(x1)
    x1 = cbind(rep(1,m),x1)  
  
    theta_trained = gradient_descent_convergence_criteria(theta=initialize_param(x,theta), x=x, y=y,iterations=iterations,convergence_criteria=convergence_criteria)
    theta_trained = as.matrix(theta_trained)
    
    return (log(1+exp(y1*(x1 %*% t(theta_trained)))))
}

log_likelihood_test_0_1_set5 = sapply(1:5,log_likelihood,x=percentage_subset(5,train_0_1_transpose),y=percentage_subset(5,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(5,test_0_1_transpose),y1=percentage_subset(5,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set5)
log_likelihood_test_0_1_set10 = sapply(1:5,log_likelihood,x=percentage_subset(10,train_0_1_transpose),y=percentage_subset(10,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(10,test_0_1_transpose),y1=percentage_subset(10,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set10)
log_likelihood_test_0_1_set15 = sapply(1:5,log_likelihood,x=percentage_subset(15,train_0_1_transpose),y=percentage_subset(15,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(15,test_0_1_transpose),y1=percentage_subset(15,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set15)
log_likelihood_test_0_1_set20 = sapply(1:5,log_likelihood,x=percentage_subset(20,train_0_1_transpose),y=percentage_subset(20,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(20,test_0_1_transpose),y1=percentage_subset(20,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set20)
log_likelihood_test_0_1_set25 = sapply(1:5,log_likelihood,x=percentage_subset(25,train_0_1_transpose),y=percentage_subset(25,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(25,test_0_1_transpose),y1=percentage_subset(25,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set25)
log_likelihood_test_0_1_set30 = sapply(1:5,log_likelihood,x=percentage_subset(30,train_0_1_transpose),y=percentage_subset(30,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(30,test_0_1_transpose),y1=percentage_subset(30,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set30)
log_likelihood_test_0_1_set35 = sapply(1:5,log_likelihood,x=percentage_subset(35,train_0_1_transpose),y=percentage_subset(35,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(35,test_0_1_transpose),y1=percentage_subset(35,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set35)
log_likelihood_test_0_1_set40 = sapply(1:5,log_likelihood,x=percentage_subset(40,train_0_1_transpose),y=percentage_subset(40,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(40,test_0_1_transpose),y1=percentage_subset(40,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set40)
log_likelihood_test_0_1_set45 = sapply(1:5,log_likelihood,x=percentage_subset(45,train_0_1_transpose),y=percentage_subset(45,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(45,test_0_1_transpose),y1=percentage_subset(45,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set45)
log_likelihood_test_0_1_set50 = sapply(1:5,log_likelihood,x=percentage_subset(50,train_0_1_transpose),y=percentage_subset(50,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(50,test_0_1_transpose),y1=percentage_subset(50,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set50)
log_likelihood_test_0_1_set55 = sapply(1:5,log_likelihood,x=percentage_subset(55,train_0_1_transpose),y=percentage_subset(55,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(55,test_0_1_transpose),y1=percentage_subset(55,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set55)
log_likelihood_test_0_1_set60 = sapply(1:5,log_likelihood,x=percentage_subset(60,train_0_1_transpose),y=percentage_subset(60,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(60,test_0_1_transpose),y1=percentage_subset(60,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set60)
log_likelihood_test_0_1_set65 = sapply(1:5,log_likelihood,x=percentage_subset(65,train_0_1_transpose),y=percentage_subset(65,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(65,test_0_1_transpose),y1=percentage_subset(65,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set65)
log_likelihood_test_0_1_set70 = sapply(1:5,log_likelihood,x=percentage_subset(70,train_0_1_transpose),y=percentage_subset(70,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(70,test_0_1_transpose),y1=percentage_subset(70,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set70)
log_likelihood_test_0_1_set75 = sapply(1:5,log_likelihood,x=percentage_subset(75,train_0_1_transpose),y=percentage_subset(75,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(75,test_0_1_transpose),y1=percentage_subset(75,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set75)
log_likelihood_test_0_1_set80 = sapply(1:5,log_likelihood,x=percentage_subset(80,train_0_1_transpose),y=percentage_subset(80,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(80,test_0_1_transpose),y1=percentage_subset(80,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set80)
log_likelihood_test_0_1_set85 = sapply(1:5,log_likelihood,x=percentage_subset(85,train_0_1_transpose),y=percentage_subset(85,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(85,test_0_1_transpose),y1=percentage_subset(85,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set85)
log_likelihood_test_0_1_set90 = sapply(1:5,log_likelihood,x=percentage_subset(90,train_0_1_transpose),y=percentage_subset(90,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(90,test_0_1_transpose),y1=percentage_subset(90,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set90)
log_likelihood_test_0_1_set95 = sapply(1:5,log_likelihood,x=percentage_subset(95,train_0_1_transpose),y=percentage_subset(95,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(95,test_0_1_transpose),y1=percentage_subset(95,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set95)
log_likelihood_test_0_1_set100 = sapply(1:5,log_likelihood,x=percentage_subset(100,train_0_1_transpose),y=percentage_subset(100,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(100,test_0_1_transpose),y1=percentage_subset(100,true_label_test_0_1_transpose))
#mean(log_likelihood_test_0_1_set100)

log_likelihood_test_0_1_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.12303","53.2387","53.73291","55.63617","61.98738")

log_likelihood_test_3_5_set5 = sapply(1:5,log_likelihood,x=percentage_subset(5,train_3_5_transpose),y=percentage_subset(5,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(5,test_3_5_transpose),y1=percentage_subset(5,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set5)
log_likelihood_test_3_5_set10 = sapply(1:5,log_likelihood,x=percentage_subset(10,train_3_5_transpose),y=percentage_subset(10,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(10,test_3_5_transpose),y1=percentage_subset(10,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set10)
log_likelihood_test_3_5_set15 = sapply(1:5,log_likelihood,x=percentage_subset(15,train_3_5_transpose),y=percentage_subset(15,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(15,test_3_5_transpose),y1=percentage_subset(15,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set15)
log_likelihood_test_3_5_set20 = sapply(1:5,log_likelihood,x=percentage_subset(20,train_3_5_transpose),y=percentage_subset(20,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(20,test_3_5_transpose),y1=percentage_subset(20,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set20)
log_likelihood_test_3_5_set25 = sapply(1:5,log_likelihood,x=percentage_subset(25,train_3_5_transpose),y=percentage_subset(25,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(25,test_3_5_transpose),y1=percentage_subset(25,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set25)
log_likelihood_test_3_5_set30 = sapply(1:5,log_likelihood,x=percentage_subset(30,train_3_5_transpose),y=percentage_subset(30,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(30,test_3_5_transpose),y1=percentage_subset(30,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set30)
log_likelihood_test_3_5_set35 = sapply(1:5,log_likelihood,x=percentage_subset(35,train_3_5_transpose),y=percentage_subset(35,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(35,test_3_5_transpose),y1=percentage_subset(35,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set35)
log_likelihood_test_3_5_set40 = sapply(1:5,log_likelihood,x=percentage_subset(40,train_3_5_transpose),y=percentage_subset(40,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(40,test_3_5_transpose),y1=percentage_subset(40,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set40)
log_likelihood_test_3_5_set45 = sapply(1:5,log_likelihood,x=percentage_subset(45,train_3_5_transpose),y=percentage_subset(45,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(45,test_3_5_transpose),y1=percentage_subset(45,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set45)
log_likelihood_test_3_5_set50 = sapply(1:5,log_likelihood,x=percentage_subset(50,train_3_5_transpose),y=percentage_subset(50,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(50,test_3_5_transpose),y1=percentage_subset(50,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set50)
log_likelihood_test_3_5_set55 = sapply(1:5,log_likelihood,x=percentage_subset(55,train_3_5_transpose),y=percentage_subset(55,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(55,test_3_5_transpose),y1=percentage_subset(55,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set55)
log_likelihood_test_3_5_set60 = sapply(1:5,log_likelihood,x=percentage_subset(60,train_3_5_transpose),y=percentage_subset(60,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(60,test_3_5_transpose),y1=percentage_subset(60,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set60)
log_likelihood_test_3_5_set65 = sapply(1:5,log_likelihood,x=percentage_subset(65,train_3_5_transpose),y=percentage_subset(65,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(65,test_3_5_transpose),y1=percentage_subset(65,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set65)
log_likelihood_test_3_5_set70 = sapply(1:5,log_likelihood,x=percentage_subset(70,train_3_5_transpose),y=percentage_subset(70,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(70,test_3_5_transpose),y1=percentage_subset(70,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set70)
log_likelihood_test_3_5_set75 = sapply(1:5,log_likelihood,x=percentage_subset(75,train_3_5_transpose),y=percentage_subset(75,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(75,test_3_5_transpose),y1=percentage_subset(75,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set75)
log_likelihood_test_3_5_set80 = sapply(1:5,log_likelihood,x=percentage_subset(80,train_3_5_transpose),y=percentage_subset(80,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(80,test_3_5_transpose),y1=percentage_subset(80,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set80)
log_likelihood_test_3_5_set85 = sapply(1:5,log_likelihood,x=percentage_subset(85,train_3_5_transpose),y=percentage_subset(85,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(85,test_3_5_transpose),y1=percentage_subset(85,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set85)
log_likelihood_test_3_5_set90 = sapply(1:5,log_likelihood,x=percentage_subset(90,train_3_5_transpose),y=percentage_subset(90,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(90,test_3_5_transpose),y1=percentage_subset(90,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set90)
log_likelihood_test_3_5_set95 = sapply(1:5,log_likelihood,x=percentage_subset(95,train_3_5_transpose),y=percentage_subset(95,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(95,test_3_5_transpose),y1=percentage_subset(95,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set95)
log_likelihood_test_3_5_set100 = sapply(1:5,log_likelihood,x=percentage_subset(100,train_3_5_transpose),y=percentage_subset(100,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(100,test_3_5_transpose),y1=percentage_subset(100,y_test_3_5_transpose))
#mean(log_likelihood_test_3_5_set100)

log_likelihood_test_3_5_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.12303","53.2387","53.73291","55.63617","61.98738")

log_likelihood_train_0_1_set5 = sapply(1:5,log_likelihood,x=percentage_subset(5,train_0_1_transpose),y=percentage_subset(5,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(5,train_0_1_transpose),y1=percentage_subset(5,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set5)
log_likelihood_train_0_1_set10 = sapply(1:5,log_likelihood,x=percentage_subset(10,train_0_1_transpose),y=percentage_subset(10,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(10,train_0_1_transpose),y1=percentage_subset(10,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set10)
log_likelihood_train_0_1_set15 = sapply(1:5,log_likelihood,x=percentage_subset(15,train_0_1_transpose),y=percentage_subset(15,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(15,train_0_1_transpose),y1=percentage_subset(15,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set15)
log_likelihood_train_0_1_set20 = sapply(1:5,log_likelihood,x=percentage_subset(20,train_0_1_transpose),y=percentage_subset(20,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(20,train_0_1_transpose),y1=percentage_subset(20,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set20)
log_likelihood_train_0_1_set25 = sapply(1:5,log_likelihood,x=percentage_subset(25,train_0_1_transpose),y=percentage_subset(25,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(25,train_0_1_transpose),y1=percentage_subset(25,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set25)
log_likelihood_train_0_1_set30 = sapply(1:5,log_likelihood,x=percentage_subset(30,train_0_1_transpose),y=percentage_subset(30,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(30,train_0_1_transpose),y1=percentage_subset(30,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set30)
log_likelihood_train_0_1_set35 = sapply(1:5,log_likelihood,x=percentage_subset(35,train_0_1_transpose),y=percentage_subset(35,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(35,train_0_1_transpose),y1=percentage_subset(35,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set35)
log_likelihood_train_0_1_set40 = sapply(1:5,log_likelihood,x=percentage_subset(40,train_0_1_transpose),y=percentage_subset(40,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(40,train_0_1_transpose),y1=percentage_subset(40,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set40)
log_likelihood_train_0_1_set45 = sapply(1:5,log_likelihood,x=percentage_subset(45,train_0_1_transpose),y=percentage_subset(45,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(45,train_0_1_transpose),y1=percentage_subset(45,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set45)
log_likelihood_train_0_1_set50 = sapply(1:5,log_likelihood,x=percentage_subset(50,train_0_1_transpose),y=percentage_subset(50,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(50,train_0_1_transpose),y1=percentage_subset(50,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set50)
log_likelihood_train_0_1_set55 = sapply(1:5,log_likelihood,x=percentage_subset(55,train_0_1_transpose),y=percentage_subset(55,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(55,train_0_1_transpose),y1=percentage_subset(55,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set55)
log_likelihood_train_0_1_set60 = sapply(1:5,log_likelihood,x=percentage_subset(60,train_0_1_transpose),y=percentage_subset(60,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(60,train_0_1_transpose),y1=percentage_subset(60,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set60)
log_likelihood_train_0_1_set65 = sapply(1:5,log_likelihood,x=percentage_subset(65,train_0_1_transpose),y=percentage_subset(65,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(65,train_0_1_transpose),y1=percentage_subset(65,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set65)
log_likelihood_train_0_1_set70 = sapply(1:5,log_likelihood,x=percentage_subset(70,train_0_1_transpose),y=percentage_subset(70,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(70,train_0_1_transpose),y1=percentage_subset(70,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set70)
log_likelihood_train_0_1_set75 = sapply(1:5,log_likelihood,x=percentage_subset(75,train_0_1_transpose),y=percentage_subset(75,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(75,train_0_1_transpose),y1=percentage_subset(75,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set75)
log_likelihood_train_0_1_set80 = sapply(1:5,log_likelihood,x=percentage_subset(80,train_0_1_transpose),y=percentage_subset(80,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(80,train_0_1_transpose),y1=percentage_subset(80,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set80)
log_likelihood_train_0_1_set85 = sapply(1:5,log_likelihood,x=percentage_subset(85,train_0_1_transpose),y=percentage_subset(85,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(85,train_0_1_transpose),y1=percentage_subset(85,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set85)
log_likelihood_train_0_1_set90 = sapply(1:5,log_likelihood,x=percentage_subset(90,train_0_1_transpose),y=percentage_subset(90,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(90,train_0_1_transpose),y1=percentage_subset(90,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set90)
log_likelihood_train_0_1_set95 = sapply(1:5,log_likelihood,x=percentage_subset(95,train_0_1_transpose),y=percentage_subset(95,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(95,train_0_1_transpose),y1=percentage_subset(95,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set95)
log_likelihood_train_0_1_set100 = sapply(1:5,log_likelihood,x=percentage_subset(100,train_0_1_transpose),y=percentage_subset(100,true_label_train_0_1_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(100,train_0_1_transpose),y1=percentage_subset(100,true_label_train_0_1_transpose))
#mean(log_likelihood_train_0_1_set100)

log_likelihood_train_0_1_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.12303","53.2387","53.73291","55.63617","61.98738")

log_likelihood_train_3_5_set5 = sapply(1:5,log_likelihood,x=percentage_subset(5,train_3_5_transpose),y=percentage_subset(5,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(5,train_3_5_transpose),y1=percentage_subset(5,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set5) = 0.6931472
log_likelihood_train_3_5_set10 = sapply(1:5,log_likelihood,x=percentage_subset(10,train_3_5_transpose),y=percentage_subset(10,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(10,train_3_5_transpose),y1=percentage_subset(10,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set10)
log_likelihood_train_3_5_set15 = sapply(1:5,log_likelihood,x=percentage_subset(15,train_3_5_transpose),y=percentage_subset(15,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(15,train_3_5_transpose),y1=percentage_subset(15,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set15)
log_likelihood_train_3_5_set20 = sapply(1:5,log_likelihood,x=percentage_subset(20,train_3_5_transpose),y=percentage_subset(20,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(20,train_3_5_transpose),y1=percentage_subset(20,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set20)
log_likelihood_train_3_5_set25 = sapply(1:5,log_likelihood,x=percentage_subset(25,train_3_5_transpose),y=percentage_subset(25,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(25,train_3_5_transpose),y1=percentage_subset(25,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set25)
log_likelihood_train_3_5_set30 = sapply(1:5,log_likelihood,x=percentage_subset(30,train_3_5_transpose),y=percentage_subset(30,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(30,train_3_5_transpose),y1=percentage_subset(30,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set30)
log_likelihood_train_3_5_set35 = sapply(1:5,log_likelihood,x=percentage_subset(35,train_3_5_transpose),y=percentage_subset(35,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(35,train_3_5_transpose),y1=percentage_subset(35,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set35)
log_likelihood_train_3_5_set40 = sapply(1:5,log_likelihood,x=percentage_subset(40,train_3_5_transpose),y=percentage_subset(40,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(40,train_3_5_transpose),y1=percentage_subset(40,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set40)
log_likelihood_train_3_5_set45 = sapply(1:5,log_likelihood,x=percentage_subset(45,train_3_5_transpose),y=percentage_subset(45,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(45,train_3_5_transpose),y1=percentage_subset(45,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set45)
log_likelihood_train_3_5_set50 = sapply(1:5,log_likelihood,x=percentage_subset(50,train_3_5_transpose),y=percentage_subset(50,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(50,train_3_5_transpose),y1=percentage_subset(50,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set50)
log_likelihood_train_3_5_set55 = sapply(1:5,log_likelihood,x=percentage_subset(55,train_3_5_transpose),y=percentage_subset(55,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(55,train_3_5_transpose),y1=percentage_subset(55,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set55)
log_likelihood_train_3_5_set60 = sapply(1:5,log_likelihood,x=percentage_subset(60,train_3_5_transpose),y=percentage_subset(60,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(60,train_3_5_transpose),y1=percentage_subset(60,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set60)
log_likelihood_train_3_5_set65 = sapply(1:5,log_likelihood,x=percentage_subset(65,train_3_5_transpose),y=percentage_subset(65,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(65,train_3_5_transpose),y1=percentage_subset(65,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set65)
log_likelihood_train_3_5_set70 = sapply(1:5,log_likelihood,x=percentage_subset(70,train_3_5_transpose),y=percentage_subset(70,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(70,train_3_5_transpose),y1=percentage_subset(70,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set70)
log_likelihood_train_3_5_set75 = sapply(1:5,log_likelihood,x=percentage_subset(75,train_3_5_transpose),y=percentage_subset(75,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(75,train_3_5_transpose),y1=percentage_subset(75,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set75)
log_likelihood_train_3_5_set80 = sapply(1:5,log_likelihood,x=percentage_subset(80,train_3_5_transpose),y=percentage_subset(80,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(80,train_3_5_transpose),y1=percentage_subset(80,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set80)
log_likelihood_train_3_5_set85 = sapply(1:5,log_likelihood,x=percentage_subset(85,train_3_5_transpose),y=percentage_subset(85,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(85,train_3_5_transpose),y1=percentage_subset(85,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set85)
log_likelihood_train_3_5_set90 = sapply(1:5,log_likelihood,x=percentage_subset(90,train_3_5_transpose),y=percentage_subset(90,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(90,train_3_5_transpose),y1=percentage_subset(90,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set90)
log_likelihood_train_3_5_set95 = sapply(1:5,log_likelihood,x=percentage_subset(95,train_3_5_transpose),y=percentage_subset(95,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(95,train_3_5_transpose),y1=percentage_subset(95,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set95)
log_likelihood_train_3_5_set100 = sapply(1:5,log_likelihood,x=percentage_subset(100,train_3_5_transpose),y=percentage_subset(100,y_train_3_5_transpose),theta=0,iterations=iterations,convergence_criteria=1,x1=percentage_subset(100,train_3_5_transpose),y1=percentage_subset(100,y_train_3_5_transpose))
#mean(log_likelihood_train_3_5_set100)

log_likelihood_train_3_5_set = c("53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.102","53.12303","53.2387","53.73291","55.63617","61.98738")

qplot(x=log_likelihood_test_3_5_set, y=log_likelihood_train_3_5_set,geom="point",xlab="likelihood of test",ylab ="likelihood of train", main="test vs train for classification 3_5")
qplot(x=log_likelihood_test_0_1_set, y=log_likelihood_train_0_1_set,geom="point",xlab="likelihood of test",ylab ="likelihood of train", main="test vs train for classification 0_1")

