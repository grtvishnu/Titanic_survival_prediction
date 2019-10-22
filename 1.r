setwd("C:/Users/optra/Downloads/tit")

titanic.train <- read.csv(file = "train.csv", stringsAsFactors = FALSE, header= TRUE)
titanic.test <- read.csv(file = "test.csv", stringsAsFactors = FALSE, header= TRUE)


titanic.train$Its <- TRUE
titanic.test$Its <- FALSE

titanic.test$Survived <- NA

titanic.full <- rbind(titanic.train, titanic.test)

titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'

#clean missing value of Age
age.median <- median(titanic.full$Age, na.rm = TRUE)
titanic.full[is.na(titanic.full$Age), "Age"] <- age.median

#clean missing value of fare
table(is.na(titanic.full$Fare))
Fare.median <- median(titanic.full$Fare, na.rm = TRUE)
titanic.full[is.na(titanic.full$Fare), "Fare"] <- Fare.median

#categorical Casting
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)




#return to train and test data
titanic.train <- titanic.full[titanic.full$Its== TRUE,]
titanic.test <- titanic.full[titanic.full$Its== FALSE,]

titanic.train$Survived<-as.factor(titanic.train$Survived)

Survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived.formula <- as.formula(Survived.equation)

install.packages("randomForest")
library(randomForest)


titanic.model<- randomForest (formula = Survived.formula, data = titanic.train, ntree = 500, mtry = 3, nodesize = 0.01 * nrow(titanic.test) )
features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
Survived <- predict(titanic.model, newdata = titanic.test)

PassengerId<- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

write.csv(output.df, file = "output.csv", row.names = FALSE)
