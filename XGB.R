train <- read.csv('train.csv',stringsAsFactors=F)
test <- read.csv('test.csv',stringsAsFactors=F)
test$Survived <- NA; data <- rbind(train,test)
# engineer person type
data$Title <- 'man'
data$Title[grep('Master',data$Name)] <- 'boy'
data$Title[data$Sex=='female'] <- 'woman'
# impute missing Age and Fare
library(rpart)
fit <- rpart(Age ~ Title + Pclass + SibSp + Parch,data=data)
data$Age[is.na(data$Age)] <- predict(fit,newdata=data[is.na(data$Age),])
fit <- rpart(Fare ~ Title + Pclass + Embarked + Sex + Age,data=data)
data$Fare[is.na(data$Fare)] <- predict(fit,newdata=data[is.na(data$Fare),])
# engineer features
data$TicketFreq <- ave(1:1309,data$Ticket,FUN=length)
data$FareAdj <- data$Fare / data$TicketFreq
data$FamilySize <- data$SibSp + data$Parch + 1
# isolate training set males
data2 <- data[data$PassengerId<=891 & data$Title=='man',]
# engineer 2 features from Fare, Ticket, SibSp, Parch, and Age
data3 <- data.frame(
        y=data2$Survived,
        x1=data2$Fare / (data2$TicketFreq * 10),
        x2=(data2$SibSp + data2$Parch + 1) + (data2$Age / 70),
        Pclass=data2$Pclass)
# plot engineered features
library(ggplot2)
library(gridExtra)
g1 = ggplot(data[data$FareAdj>0 & data$FareAdj<40,]) + 
        geom_density(aes(x=FareAdj,fill=factor(Pclass)),alpha=0.9,show.legend=F) +
        labs(fill='Pclass') + geom_vline(xintercept=c(10,20),linetype='dotted') +
        xlim(0,40) + labs(title='All Passengers',x='x1 = Fare / Ticket Frequency')
g2 = ggplot(data[data$Fare>0 & data$Fare<40 ,]) +
        geom_density(aes(x=Fare,fill=factor(Pclass)),alpha=0.9) +
        xlim(0,40) + labs(title='All Passengers',fill='Pclass',y='')
g3 = ggplot(data[!is.na(data$Survived) & data$Title=='man' & data$Pclass==1 & data$FamilySize<4,]) + 
        geom_density(aes(x=FamilySize+Age/70,fill=factor(Survived)),alpha=0.9,bw=0.04) +
        labs(fill='Survived') +
        annotate('text',x=1.4,y=2.4,label='Age 30\nFS=1') +
        annotate('text',x=2.4,y=2.4,label='Age 30\nFS=2') +
        geom_vline(xintercept=c(1.43,2.43),linetype='dotted') +
        xlim(1,4) + labs(title='Adult Males Pclass=1',x='x2 = FamilySize + Age/70')
grid.arrange(g3,g1,g2,layout_matrix = rbind(c(1, 1), c(2, 3)))










# classify males with xgboost
library(xgboost)
x1s <- seq(0, 5, length.out = 100)
x2s <- seq(1, 3, length.out = 100)
g <- data.frame(x1 = rep(x1s, each=100), x2 = rep(x2s, time = 100))
param <- list(objective   = "binary:logistic",
              eval_metric = "error",
              max_depth   = 5,
              eta         = 0.1,
              gammma      = 0.1,
              colsample_bytree = 1,
              min_child_weight = 1)
cat('XGBoosting begun...\n')
xgb <- xgboost(params  = param,
               data    = data.matrix(data3[,c('x1','x2')]),
               label   = data3$y, 
               nrounds = 500,
               print_every_n = 100,
               verbose = 1)
p <- predict(xgb,newdata=data.matrix(g[,c('x1','x2')]))
g$y <- ifelse(p>0.5,1,0)
# plot classification regions
ggplot(data=data3[data3$x1<5 & data3$x2<3,]) +
        xlim(0,5) + ylim(1,3) +
        geom_tile(data=g,aes(x1,x2,fill=factor(y))) +
        geom_density_2d(aes(x1,x2,color=factor(y))) +
        geom_point(size=2,aes(x1,x2,color=factor(y),shape=factor(Pclass))) +
        scale_color_manual(values=c('#AA0000','#00AA00'),
                           limits=c('0','1'),labels=c('0','1')) +
        scale_fill_manual(values=c('#FF9999','#99FF99'),
                          limits=c('0','1'),labels=c('0','1')) +
        labs(x='Fare / (10 x TicketFrequency)',y='FamilySize + (Age / 70)',shape='Pclass',fill='Classify',
             title='XGBoost learns the training set\'s
        537 adult males. Green is P(live)>0.5',color='Survived') +
        geom_vline(xintercept=2.8, linetype='dotted') +
        geom_hline(yintercept=c(1.43,2.43), linetype='dotted') +
        annotate('text',x=2.95,y=2.9,label='Fare = $28') +
        annotate('text',x=4.7,y=2.35,label='Age = 30') +
        annotate('text',x=4.7,y=1.35,label='Age = 30')
         







set.seed(300)
s = sample(1:537,100)
s2 = (1:537)[-s]
xgb <- xgboost(params  = param,
               data    = data.matrix(data3[s2,c('x1','x2')]),
               label   = data3$y[s2], 
               nrounds = 500,
               print_every_n = 100,
               silent = 1,
               verbose = 0)
p <- predict(xgb,newdata=data.matrix(data3[s,c('x1','x2')]))
data3$y2 <- NA
roc = data.frame(TN=rep(0,102),FP=rep(0,102),FN=rep(0,102)
                 ,TP=rep(0,102),FPR=rep(0,102),TPR=rep(0,102))
rownames(roc) <- seq(0,1.01,0.01)
for (i in 1:102){
        data3$y2[s] <- ifelse(p<(i-1)/100,0,1)
        roc$TP[i] <- length(which(data3$y==1 & data3$y2==1))
        roc$TN[i] <- length(which(data3$y==0 & data3$y2==0))
        roc$FP[i] <- length(which(data3$y==0 & data3$y2==1))
        roc$FN[i] <- length(which(data3$y==1 & data3$y2==0))
        roc$FPR[i] <- roc$FP[i] / (roc$FP[i] + roc$TN[i])
        roc$TPR[i] <- roc$TP[i] / (roc$TP[i] + roc$FN[i])
}
diag <- data.frame(x=c(0,1),y=c(0,1))
ggplot(roc,aes(x=FPR,y=TPR)) + 
        geom_line() + labs(title='ROC curve where "positive" = male survives',
                           x='False positive rate',y='True positive rate') +
        geom_point(data=roc[91,],aes(x=FPR,y=TPR),size=3,color='red') +
        annotate('text',x=0.13,y=0.12,label='threshold p>=0.9',color='red') +
        geom_point(data=roc[76,],aes(x=FPR,y=TPR),size=3,color='darkgreen') +
        annotate('text',x=0.16,y=0.30,label='threshold p>=0.75',color='darkgreen') +
        geom_point(data=roc[51,],aes(x=FPR,y=TPR),size=3,color='blue') +
        annotate('text',x=0.20,y=0.5,label='threshold p>=0.5',color='blue') +
        geom_line(data=diag,aes(x=x,y=y),linetype='dotted')
area = 0
for(i in 1:101){
        area = area + roc$TPR[i] * (roc$FPR[i]-roc$FPR[i+1])
}
cat(sprintf('Area under ROC = %f\n',area))












#accuracy = matrix(nrow=4,ncol=4)
#rownames(accuracy) <- c('d=3','d=4','d=5','d=6')
#colnames(accuracy) <- c('0.50','0.75','0.90','0.92')
#for (j in 1:4)
#for (k in 1:4){
set.seed(2)
trials=100; sum=0
for (i in 1:trials){
        s = sample(1:891,800)
        s2 = (1:891)[-s]
        dataB <- data[data$PassengerId %in% s & data$Title=='man',]
        dataC <- data[data$PassengerId %in% s2 & data$Title=='man',]
        data$Predict <- 0
        data$Predict[data$Sex=='female'] <- 1
        dataTrain <- data.frame(y=dataB$Survived,x1=dataB$FareAdj/10,x2=dataB$FamilySize+dataB$Age/70)
        dataTest <- data.frame(y=dataC$Survived,x1=dataC$FareAdj/10
                               ,x2=dataC$FamilySize+dataC$Age/70,PassengerId=dataC$PassengerId)
        param <- list(objective   = "binary:logistic",
                      eval_metric = "error",
                      max_depth   = 5,
                      eta         = 0.1,
                      gammma      = 0.1,
                      colsample_bytree = 1,
                      min_child_weight = 1)
        xgb <- xgboost(params  = param,
                       data    = data.matrix(dataTrain[,c('x1','x2')]),
                       label   = dataTrain$y, 
                       nrounds = 500,
                       silent = 1,
                       print_every_n = 100,
                       verbose = 0)
        p <- predict(xgb,newdata=data.matrix(dataTest[,c('x1','x2')]))
        dataTest$p <- ifelse(p>=0.90,1,0)
        data$Predict[dataTest$PassengerId] <- dataTest$p 
        sm = sum(abs(data$Predict[s2] - data$Survived[s2]))
        cv = 1 - sm/length(s2)
        #if (i %% 25==0) 
        #cat(sprintf('Trial %d has CV = %f\n',i,cv))
        sum = sum + cv
}
cat(sprintf('Average CV of %d trials = %f\n',trials,sum/trials))
# accuracy[j,k] <- sum/trials
# }
#accuracy

dataB <- data[data$PassengerId %in% 1:891 & data$Title=='man',]
dataC <- data[data$PassengerId %in% 892:1309 & data$Title=='man',]
data$Predict <- 0
data$Predict[data$Sex=='female'] <- 1
dataTrain <- data.frame(y=dataB$Survived,x1=dataB$FareAdj/10,x2=dataB$FamilySize+dataB$Age/70)
dataTest <- data.frame(y=dataC$Survived,x1=dataC$FareAdj/10,Pclass=dataC$Pclass
                       ,x2=dataC$FamilySize+dataC$Age/70,PassengerId=dataC$PassengerId)
param <- list(objective   = "binary:logistic",
              eval_metric = "error",
              max_depth   = 5,
              eta         = 0.1,
              gammma      = 0.1,
              colsample_bytree = 1,
              min_child_weight = 1)
xgb <- xgboost(params  = param,
               data    = data.matrix(dataTrain[,c('x1','x2')]),
               label   = dataTrain$y, 
               nrounds = 500,
               silent = 1,
               print_every_n = 100,
               verbose = 0)
p <- predict(xgb,newdata=data.matrix(dataTest[,c('x1','x2')]))
dataTest$p <- ifelse(p>=0.90,1,0)
data$Predict[dataTest$PassengerId] <- dataTest$p 
maleLive = which(data$Title=='man' & data$Predict==1)
cat(sprintf('The following %d adult males are predicted to live\n',length(maleLive)))
data[maleLive,c('PassengerId','Pclass','Name','Age','SibSp','Parch','FareAdj')]




x1s <- seq(2, 5, length.out = 100)
x2s <- seq(1, 3, length.out = 100)
g <- data.frame(x1 = rep(x1s, each=100), x2 = rep(x2s, time = 100))
p <- predict(xgb,newdata=data.matrix(g[,c('x1','x2')]))
g$y <- ifelse(p>=0.90,1,0)
g1 <- ggplot(data=dataTest[dataTest$x1>2 & dataTest$x1<5 & dataTest$x2>1 & dataTest$x2<3,]) +
        xlim(2,5) + ylim(1,3) +
        geom_tile(data=g,aes(x1,x2,fill=factor(y))) +
        geom_point(size=2,aes(x1,x2,color=factor(p),shape=factor(Pclass))) +
        scale_color_manual(values=c('#666666','#0000FF'),
                           limits=c('0','1'),labels=c('0','1')) +
        scale_fill_manual(values=c('#FF9999','#99FF99'),
                          limits=c('0','1'),labels=c('0','1')) +
        labs(x='Fare / (10 x TicketFrequency)',y='FamilySize + (Age / 70)',shape='Pclass',fill='Classifier',
             title='XGBoost classifies the test set.
        It predicts 4 adult males have P(live)>=0.9',color='Predict') +
        geom_vline(xintercept=2.8, linetype='dotted') +
        geom_hline(yintercept=c(1.43,2.43), linetype='dotted') +
        annotate('text',x=2.95,y=2.9,label='Fare = $28') +
        annotate('text',x=4.7,y=2.35,label='Age = 30') +
        annotate('text',x=4.7,y=1.35,label='Age = 30')

for (i in which(dataTest$p==1)){
        g1 <- g1 + annotate('text',x=dataTest$x1[i]-0.15,y=dataTest$x2[i],label=dataTest$PassengerId[i]
                            ,color='darkblue',size=4)
}
g1







# engineer "woman-child-groups"
data$Surname = substring( data$Name,0,regexpr(',',data$Name)-1)
data$GroupId = paste( data$Surname, data$Pclass, sub('.$','X',data$Ticket), data$Fare, data$Embarked, sep='-')
data$GroupId[data$Title=='man'] <- 'noGroup'
# Mrs Wilkes (Needs) is Mrs Hocking (Needs) sister
data$GroupId[893] <- data$GroupId[775]
data$GroupFreq <- ave(1:1309,data$GroupId,FUN=length)
data$GroupId[data$GroupFreq<=1] <- 'noGroup'
cat(sprintf('We found %d woman-child-groups\n',length(unique(data$GroupId))-1))
data$TicketId = paste( data$Pclass,sub('.$','X',data$Ticket),data$Fare,data$Embarked,sep='-')
# add nannies and relatives to groups
c = 0
for (i in which(data$Title!='man' & data$GroupId=='noGroup')){
        data$GroupId[i] = data$GroupId[data$TicketId==data$TicketId[i]][1]
        if (data$GroupId[i]!='noGroup') c = c + 1
}
cat(sprintf('We found %d nannies/relatives and added them to groups.\n',c))
# calculate group survival rates
data$GroupSurvival <- NA
data$Survived <- as.numeric(as.character(data$Survived))
data$GroupSurvival[1:891] <- ave(data$Survived[1:891],data$GroupId[1:891])
for (i in 892:1309) data$GroupSurvival[i] <- data$GroupSurvival[which(data$GroupId==data$GroupId[i])[1]]
# classify unknown groups
data$GroupSurvival[is.na(data$GroupSurvival) & data$Pclass==3] <- 0
data$GroupSurvival[is.na(data$GroupSurvival) & data$Pclass!=3] <- 1
# make predictions
data$Predict <- 0
data$Predict[data$Sex=='female'] <- 1
data$Predict[data$Title=='woman' & data$GroupSurvival==0] <- 0
data$Predict[data$Title=='boy' & data$GroupSurvival==1] <- 1
x = which(data$Sex=='male' & data$Predict==1 & data$PassengerId>891)
cat(sprintf('We found %d boys predicted to live\n',length(x)))
x = which(data$Sex=='female' & data$Predict==0 & data$PassengerId>891)
cat(sprintf('We found %d females predicted to die\n',length(x)))


submit <- data.frame(PassengerId=892:1309,Survived=data$Predict[892:1309])
submit$Survived[maleLive-891] <- 1
write.csv(submit,'WCG_XGBoost1.csv',row.names=F)


