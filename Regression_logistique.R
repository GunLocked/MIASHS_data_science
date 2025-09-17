library(dplyr)
library(ggplot2)
library(ROCR)



id_diff <- read.csv("/Users/mekkiryan/Desktop/m-1-miashs-1-2-journee-data-science/ex_soumission.csv", sep=",", dec=".", header=TRUE)
train   <- read.csv("/Users/mekkiryan/Desktop/m-1-miashs-1-2-journee-data-science/farms_test.csv", sep=",", dec=".", header=TRUE)


train$DIFF <- id_diff$DIFF


glimpse(train)
table(train$DIFF)


ggplot(train, aes(DIFF, R2, fill=DIFF)) + geom_boxplot()


ggplot(train, aes(DIFF, R7, fill=DIFF)) + geom_boxplot()

set.seed(123) 
n <- nrow(train)
idx <- sample(1:n, size = round(0.7*n))

train_data <- train[idx, ]     
valid_data <- train[-idx, ]     

fit_logit <- glm(DIFF ~ R2 + R7 + R8 + R17 + R22 + R32,
                 family = binomial,
                 data = train_data)

summary(fit_logit)


pred_prob <- predict(fit_logit, newdata=valid_data, type="response")


alpha <- 1/2
pred_class <- ifelse(pred_prob > alpha, 1, 0)


table(Pred = pred_class, Réel = valid_data$DIFF)



pred <- prediction(pred_prob, valid_data$DIFF)
perf <- performance(pred, "tpr", "fpr")


plot(perf, col="blue", lwd=2,
     main="Courbe ROC - modèle logistique")
abline(0,1,lty=2,col="red")


auc <- performance(pred, "auc")@y.values[[1]]
auc


pred_prob <- predict(fit_logit, newdata=valid_data, type="response")


seuils <- seq(0, 1, by=0.1)


for (s in seuils) {
  pred_class <- ifelse(pred_prob > s, 1, 0)
  cat("\n--- Seuil =", s, "---\n")
  print(table(Pred = pred_class, Réel = valid_data$DIFF))
}




vars <- c("R2", "R7", "R8", "R17", "R22", "R32")


all_combinations <- unlist(
  lapply(1:length(vars),
         function(k) combn(vars, k, simplify=FALSE)),
  recursive=FALSE
)


compute_auc <- function(var_list, train_data, valid_data){
  
  formula <- as.formula(paste("DIFF ~", paste(var_list, collapse=" + ")))
  
  fit <- glm(formula, family="binomial", data=train_data)
  
  predictions <- predict(fit, newdata=valid_data, type="response")
  
  pred <- prediction(predictions, valid_data$DIFF)
  
  auc  <- performance(pred, "auc")@y.values[[1]]
  
  return(auc)
}


results <- data.frame(
  variables = sapply(all_combinations, paste, collapse=" + "),
  AUC = sapply(all_combinations, compute_auc,
               train_data=train_data, valid_data=valid_data)
)


results <- results %>% arrange(desc(AUC))


results



fit_full <- glm(DIFF ~ R2 + R7 + R8 + R17 + R22 + R32,
                family = binomial, data = train_data)

prob_full <- predict(fit_full, newdata=valid_data, type="response")

pred_full <- prediction(prob_full, valid_data$DIFF)
perf_full <- performance(pred_full, "tpr", "fpr")
auc_full  <- performance(pred_full, "auc")@y.values[[1]]


fit_best <- glm(DIFF ~ R7 + R17,
                family = binomial, data = train_data)

prob_best <- predict(fit_best, newdata=valid_data, type="response")

pred_best <- prediction(prob_best, valid_data$DIFF)
perf_best <- performance(pred_best, "tpr", "fpr")
auc_best  <- performance(pred_best, "auc")@y.values[[1]]



dev.new()


plot(perf_full, col="blue", lwd=2,
     main="Courbes ROC - Comparaison modèles")


plot(perf_best, col="red", lwd=2, add=TRUE)


abline(0,1,lty=2,col="gray")


legend("bottomright",
       legend=c(paste("Complet - AUC =", round(auc_full,3)),
                paste("R7+R17 - AUC =", round(auc_best,3))),
       col=c("blue","red"), lwd=2)


prob_full <- predict(fit_full, newdata=valid_data, type="response")
pred_full_class <- ifelse(prob_full > 0.5, 1, 0)

cm_full <- table(Pred = pred_full_class, Réel = valid_data$DIFF)
cat("\nMatrice de confusion - Modèle complet\n")
print(cm_full)


prob_best <- predict(fit_best, newdata=valid_data, type="response")
pred_best_class <- ifelse(prob_best > 0.5, 1, 0)

cm_best <- table(Pred = pred_best_class, Réel = valid_data$DIFF)
cat("\nMatrice de confusion - Modèle R7+R17\n")
print(cm_best)

compute_metrics <- function(cm){
  accuracy <- sum(diag(cm)) / sum(cm)
  sensitivity <- cm["1","1"] / sum(cm[,"1"])
  specificity <- cm["0","0"] / sum(cm[,"0"])
  
  return(c(Accuracy=accuracy, Sensibilité=sensitivity, Spécificité=specificity))
}

cat("\n--- Modèle complet ---\n")
print(compute_metrics(cm_full))

cat("\n--- Modèle R7+R17 ---\n")
print(compute_metrics(cm_best))

prop.table(table(train_data$DIFF))
prop.table(table(valid_data$DIFF))
