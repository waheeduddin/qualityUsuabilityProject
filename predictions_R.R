splitdf <- function(dataframe, seed=NULL) {
	if (!is.null(seed)) set.seed(seed)
	index <- 1:nrow(dataframe)
	trainindex <- sample(index, trunc(3 * length(index)/4))
	trainset <- dataframe[trainindex, ]
	testset <- dataframe[-trainindex, ]
	list(trainset=trainset,testset=testset)
}
function putLabels(_x){
	if(as.character(_x) == "N"){
		return FALSE
	}
	if(as.character(_x) == "n"){
		return FALSE
	}
	if(as.character(_x) == "Y"){
		return TRUE
	}
	if(as.character(_x) == "y"){
		return TRUE
	}
	return FALSE
}

indivWorker = read.csv(file="full_data_small_waheed_acceptance2.csv", header=TRUE, sep=",")
#converting to right Factors and numbers for RF moel and GLM model respectively	
indivWorker$ACCEPTANCE_labels <-sapply(indivWorker$ACCEPTANCE , putLabels(x))
indivWorker$ACCEPTANCE_numeric <- rep(0, nrow(indivWorker))
indivWorker$ACCEPTANCE_numeric[which(as.character(indivWorker$ACCEPTANCE_labels) == TRUE)] = 1

#Random Forest predictions
library(randomForest)
splits <- splitdf(indivWorker, seed=808)
training <- splits$trainset
testing <- splits$testset
output.forest <- randomForest(ACCEPTANCE_labels ~FREQUENCY +Intresting + OverallEffort+ reward_div_workload, 
           data = training)
print(output.forest) 
varImpPlot(output.forest)
predicted <- predict(output.forest, newdata=testing[ ,-1])

errors <- 0
for(z in c(1:length(predicted))){
	if(predicted[z] != testing$ACCEPTANCE_labels[z])
		errors <- errors + 1
}

cat("errors: ",errors ,"/",length(predicted),"\n")

precision <- sum(predicted & testing$ACCEPTANCE_labels) / sum(predicted)
recall <- sum(predicted & testing$ACCEPTANCE_labels) / sum(testing$ACCEPTANCE_labels)
fmeasure <- 2 * precision * recall / (precision + recall)
#http://blog.datadive.net/interpreting-random-forests/
preds = matrix(as.character(predicted), ncol=1)
refs = matrix(as.character(testing$ACCEPTANCE_labels), ncol=1)
table(refs, preds)

##########GLM predictions
myLogit <- glm(ACCEPTANCE_numeric ~ reward_div_workload + Intresting + FREQUENCY + OverallEffort_mean, 
           data = training, family = "binomial")
predicted <- predict(myLogit, newdata = testing, type = "response")
errors <- 0
newPredicted <- apply(predicted , 1 function(x) {ifelse( x > 0.5 , 1 , 0)})

for(z in c(1:length(predicted))){
	if(newPredicted[z] != testing$myNewQtake[z])
		errors <- errors + 1
}

library(aod)
library(Rcpp)
library(ggplot2)
#HL test, overall CHi score , dependace individually, R score,accuracy and recall

library(ResourceSelection)
hl_test <- hoslem.test(training$ACCEPTANCE_numeric , fitted(myLogit) , g=10)
#w_test <- wald.test(b = coef(myLogit), Sigma = vcov(myLogit))
odds_ratios <- exp(coef(myLogit))
odds_CI <- exp(cbind(OR = coef(myLogit), confint(myLogit)))
chiSquare <- with(myLogit, null.deviance - deviance)
degOfFreedom <- with(myLogit, df.null - df.residual)
pValue <-  format(round(with(myLogit, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE)), 5), nsmall = 5)
rSquareValues <- RsqGLM(model = myLogit)
