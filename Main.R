library(tidyverse)
library(tidymodels)
library(ggthemes) 
library(scales) 
library(mice) 
library(randomForest) 

train <- read_csv("train.csv")
test <- read_csv("test.csv")

str(train)

train <- train %>% 
        mutate(IsTrain=T)

test <- test %>% 
        mutate(IsTrain=F)

ncol(train)
ncol(test)

names(train)
names(test)

test <- test %>% 
        mutate(Survived = NA)

full <- rbind(train,test)

str(full)

full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)
full$Pclass <- as.factor(full$Pclass)

summary(full)

full$Embarked <- replace_na(full$Embarked,"S")

full %>% 
        select(Age) %>% 
        is.na() %>% 
        table()


 up<- boxplot.stats(full$Fare)$stats[5]
 out <- full$Fare<up
 fare_mod <- lm(
         Fare~Pclass + Sex + Age + SibSp + Parch + Embarked,
         data = full[out,]
 )
 
fare_row<- full[is.na(full$Fare),c("Pclass","Sex", "Age","SibSp", "Parch", "Embarked")]
fare_pred<- predict(fare_mod, newdata = fare_row)
full[is.na(full$Fare), "Fare"] <- fare_pred

bench_fare <- 31.275+1.5*IQR(full$Fare)
full$Fare[full$Fare>bench_fare] <- bench_fare
boxplot(full$Fare)

full$Age <- replace_na(full$Age, mean(full$Age, na.rm = T))


# EDA ---------------------------------------------------------------------------------------

