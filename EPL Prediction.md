---
title: "Prediction of Premier League Performance with Expected Stats"
author: "Junghoon Kang"
date: '2021 12 13 '

---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE,message = FALSE)
```

```{r}
library(tidyverse)
library(dplyr)
library(purrr)
library(caret)
library(ggplot2)
library(kableExtra)
```



# Introduction

>Throughout the years, statistics have become a large part of sports. In baseball there is sabermetrics that started in the mid-1900s, and basketball has an APBR metric. But we've never heard of any soccer related metrics. We wanted to see if we could find metrics for soccer. However, in soccer, there are no standardized metrics like sabermetrics. This is because, unlike other sports, soccer does not produce many scores. In 2020, the average score for an MLB team per game was 4.83 runs, which is about 10 points per game, and the NBA average game score was 111.8 points which is about 200 points per game. On the other hand, soccer, EPL (English Premier League) average score per game was 2.69 which scores much lower than other sports. It can be seen that soccer has some sort of randomness, which is why soccer is more difficult to predict than other sports. In other words, baseball and basketball can collect more scoring data per game which can be generalized, but soccer, with strong randomness,  requires an index that can fairly evaluate players while excluding random elements such as luck as much as possible. However, we found the concept of "Expected Goals", which is a data in which “Opta” calculates the likelihood of scoring in a specific situation and in a specific location from data from more than 300,000 shots in 2017. We wanted to test to see how important these statistics were to earning points in the league which would consequently lead to more wins. We ran regression tests as well as KNN and Random Forest to find whether or not these stats can be considered significant or not. If we can use these statistics to see if teams will win more games, we can manage our teams more efficiently by focusing on improving xG and related stats.

## Thesis

>The 2019 EPL team’s rank(pts) predicted by the ML technique using 6 years of xG data from 6 leagues was similar to the actual rank and points. 
  


# Data Section

>We used two different 2014 -2019 6 UEFA leagues data set in our project which first data is statistical summary data of expected goal stats by the end of each season and second data is expected goal stats of each match results. Seasonal data has 684 observations with 24 variable which include 13 variable of expected goal related statistics and 11 basic game data like name of team or final rank. Each game data has 24580 observations with 29 variables 

>Our dataset was located on Kaggle which web scraping from orginal data website called Understat that produces “Expected Stats” in the soccer world. We looked at this data set that has xG/xGA which is expected goals/against, xG_diff/xGA_diff which is difference between actual goals missed and expected goals/against, npxG/npxGA which is expected goals/against without own goals and penalties, deep/deep-allowed which constitutes passes completed within an estimated 20 yards of goal, ppda_coef/oppda_coef which are passes allowed during defensive action (power of pressure). 

>We were specifically looking at stats from the English Premier league and our data source had stats from 2014-19. We were also curious to see if salary payroll was significant towards total points so we found the payrolls for the 20 teams in the EPL in 2019.

>Primary data sources link
https://understat.com/
https://www.kaggle.com/slehkyi/extended-football-stats-for-european-leagues-xg

> Process of scraping data from web portal understat.com (original data source) 
https://www.kaggle.com/slehkyi/web-scraping-football-statistics-2014-now



```{r}

stat_year=read.csv(file="understat__com.csv")
stat_per_game=read.csv(file="understat_per_game.csv")
per_game<-stat_per_game %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)
per_game<-per_game %>% relocate(result)

```

# Methods 

>Our goal is to find whether there is highly relationship between expected goal stats and rank of the season. Hence the teams that have better expected goal stats are higher rank on the season. First, Using the Seasonal data from 2014-2018 which include 6 league, we conduct linear regression model to find what statistics we could consider significant and predict 2019 season rank. 

>We used Min-Max normalization to normalize each column and have them relative to each team in the column. We chose to do this first because if we found which statistics were significant with a linear regression, we could put those into our next Machine Learning method.

>After found significant variable, using these variables we perform k Nearest Neighbors(kNN) and Random Forest method to predict 2019 season using the full game data from 2014-2019 which include 6 league. Similar with season data, we also had min-max scaling to the full game data and split the data into train and test set to 70:30. For both algorithm we used “caret” R package and construct model with 10 fold cross validation with 5 repeat to avoid bias. Also for kNN method we used euclidean distance for calculating distance between nearest neighbors.

# Results

>The results for our data can be split into two different sections. We ran the linear regression to find significance for the statistics based on full seasons. 


```{r}
library(tidyverse)

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
  

full_data <- read.csv("understat__com.csv")

prem_data <- full_data %>% 
  filter(League == "EPL")

prem_normal <- as.data.frame(lapply(prem_data[12:24], min_max_norm))

```

```{r}

prem_text = prem_data %>% select(Year,team, pts)

normalized_data = cbind(prem_text,prem_normal)

norm_prem_data_predict <- normalized_data %>% 
  filter(Year != 2019)
norm_prem_data_2019 <- normalized_data %>% 
  filter(Year == 2019)

```





## linear regression 

```{r}
lm_norm_prem_data_predict = lm(pts ~ xG + xG_diff + npxG + xGA + xGA_diff + npxGA + 
                                 npxGD + ppda_coef + oppda_coef + deep + deep_allowed  
                               ,data = norm_prem_data_predict)

summary(lm_norm_prem_data_predict)

```

>With the season data, we could find the significance of the variables between xg stat and pts using linear model.For this model, we found that the statistics that stayed consistently significant were "xG", "xG_diff" and "xGA_diff" which  p value were less than 0.05. While we did find a low p-value for some of our variables, we found an extremely high r squared value which leads us to believe that our model is still useful in predicting points. 

>We concluded that this must be due to the fact that we had a small sample size so we decided to use each game data for machine learning method kNN and Random Forest.Using these variables and xGA which xGA_diff was derived from, we will predicted the 2019 EPL rank and points by using these variables as variables for kNN and Random Forest models.




## kNN


```{r}
per_game<-stat_per_game %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)
per_game<-per_game %>% relocate(result)

idx <- sample(1:nrow(per_game), size = nrow(per_game)*0.7, replace=FALSE)
train <- per_game[idx,]
test <- per_game[-idx,]

train_mean= test[2:ncol(test)] %>% map_dbl(mean)
train_Sd= test[2:ncol(test)] %>% map_dbl(sd)


min_max_norm <- function(x) {
    (x - min(x)) / (max(x) - min(x))
  }

normal= function(x) {((x-train_mean)/train_Sd) %>% round(4)}

train[2:ncol(train)]= train[2:ncol(train)] %>% t() %>% normal(.) %>% t() %>% as_tibble()
test[2:ncol(test)]= test[2:ncol(test)] %>% t() %>% normal(.) %>% t() %>% as_tibble()
```

```{r}
ctrl= trainControl(method = "repeatedcv",
                   repeats = 5,
                   number = 10)
max_no= nrow(train) %/% 2
grid= expand.grid(k=seq(1,35,2))

knnFit= train(result ~ .,
              data= train,
              method= "knn",
              trControl= ctrl,
              tuneGrid=grid)
knnFit
```


```{r}
knnFit$results %>% ggplot(aes(x=k,y=Accuracy))+
  geom_point(color="red", size=2)+
  geom_line(color="gray", linetype=1)+
  geom_line(aes(x=k,y=Kappa),color="blue")

test1<-test %>% select(-result)
predict_test <- predict(knnFit, newdata = test1)
```


```{r}
 
predict_test_prob <- predict(knnFit, newdata = test1, type = "prob")

confusionMatrix(data = predict_test, as.factor(test$result))

```

>First, in kNN, when k is 27, the train set accuracy was 0.9916 which is highest. With k is 0.9919, the test accuracy is 0.9919, 95% Confidence interval (0.9895, 0.9938),average of Sensitivity and  Specificity also exceed 0.99, therefore, we can conclude there is meaningful result on performance of kNN. 

## Random Forest

```{r}
library(caret)
ctrl= trainControl(method = "repeatedcv",
                   repeats = 5, 
                   number = 10)
randomF= train(result ~ .,
              data= train,
              method= "rf",
              trControl= ctrl)

randomF_predict_test <- predict(randomF, newdata = test1)

confusionMatrix(data = randomF_predict_test, as.factor(test$result))
```


>However, even if the test accuracy is 0.99, the actual ranking changes due to one match, so we need higher accuracy.
So we obtained the results using the Random Forest algorithm, and the train accuracy is 1,95% Confidence interval (0.9995, 1), Sensitivity and Specificity are also 1, so our model showed almost perfect performance. 



## Prediction

>We perform three different prediction with linear regression, kNN and Random Forest.


```{r}
predict_pts_lm = predict(lm_norm_prem_data_predict,
                                     newdata = norm_prem_data_2019)

prem_data_2019 <- prem_data %>%
  filter(Year == 2019)

our_prediction = cbind(prem_data_2019,predict_pts_lm)

newdata <- our_prediction[order(-predict_pts_lm),]

comparison_data = newdata[,c(2:4,11,23,25)]


```


```{r}
## kNN 

Arsenal<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Arsenal") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

AS<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Aston Villa") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Bournemouth<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Bournemouth") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Brighton<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Brighton") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Burnley<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Burnley") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Chelsea<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Chelsea") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Crystal<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Crystal Palace") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Everton<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Everton") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Leicester<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Leicester") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Liverpool<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Liverpool") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

City<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Manchester City") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)


Manchester<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Manchester United") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Newcastle<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Newcastle United") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Norwich<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Norwich") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Sheffield<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Sheffield United") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Southampton<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Southampton") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Tottenham<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Tottenham") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Watford<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Watford") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

West<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="West Ham") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)

Wolverhampton<-stat_per_game %>% filter(league=="EPL",year=="2019",team=="Wolverhampton Wanderers") %>% select(-league,-date,-team,-year,-pts,-wins,-draws,-loses,-h_a)


Pts_K=vector()

Arsenal_random <- predict(knnFit, newdata = Arsenal)
p=0
for(i in 1:length(Arsenal_random)){
if(Arsenal_random[i]=="d"){p=p+1}
  if(Arsenal_random[i]=="w"){p=p+3}
}

Pts_K[1]<-p

AS_random <- predict(knnFit, newdata = AS)
p=0
for(i in 1:length(AS_random)){
if(AS_random[i]=="d"){p=p+1}
  if(AS_random[i]=="w"){p=p+3}
}

Pts_K[2]<-p

Bournemouth_random <- predict(knnFit, newdata = Bournemouth)
p=0
for(i in 1:length(Bournemouth_random)){
if(Bournemouth_random[i]=="d"){p=p+1}
  if(Bournemouth_random[i]=="w"){p=p+3}
}

Pts_K[3]<-p


Brighton_random <- predict(knnFit, newdata = Brighton)
p=0
for(i in 1:length(Brighton_random)){
if(Brighton_random[i]=="d"){p=p+1}
  if(Brighton_random[i]=="w"){p=p+3}
}

Pts_K[4]<-p

Burnley_random <- predict(knnFit, newdata = Burnley)
p=0
for(i in 1:length(Burnley_random)){
if(Burnley_random[i]=="d"){p=p+1}
  if(Burnley_random[i]=="w"){p=p+3}
}

Pts_K[5]<-p

Chelsea_random <- predict(knnFit, newdata = Chelsea)
p=0
for(i in 1:length(Chelsea_random)){
if(Chelsea_random[i]=="d"){p=p+1}
  if(Chelsea_random[i]=="w"){p=p+3}
}

Pts_K[6]<-p

Crystal_random <- predict(knnFit, newdata = Crystal)
p=0
for(i in 1:length(Crystal_random)){
if(Crystal_random[i]=="d"){p=p+1}
  if(Crystal_random[i]=="w"){p=p+3}
}

Pts_K[7]<-p

Everton_random <- predict(knnFit, newdata = Everton)
p=0
for(i in 1:length(Everton_random)){
if(Everton_random[i]=="d"){p=p+1}
  if(Everton_random[i]=="w"){p=p+3}
}

Pts_K[8]<-p

Leicester_random <- predict(knnFit, newdata = Leicester)
p=0
for(i in 1:length(Leicester_random)){
if(Leicester_random[i]=="d"){p=p+1}
  if(Leicester_random[i]=="w"){p=p+3}
}

Pts_K[9]<-p

Liverpool_random <- predict(knnFit, newdata = Liverpool)
p=0
for(i in 1:length(Liverpool_random)){
if(Liverpool_random[i]=="d"){p=p+1}
  if(Liverpool_random[i]=="w"){p=p+3}
}

Pts_K[10]<-p

City_random <- predict(knnFit, newdata = City)
p=0
for(i in 1:length(City_random)){
if(City_random[i]=="d"){p=p+1}
  if(City_random[i]=="w"){p=p+3}
}

Pts_K[11]<-p

Manchester_random <- predict(knnFit, newdata = Manchester)
p=0
for(i in 1:length(Manchester_random)){
if(Manchester_random[i]=="d"){p=p+1}
  if(Manchester_random[i]=="w"){p=p+3}
}

Pts_K[12]<-p

Newcastle_random <- predict(knnFit, newdata = Newcastle)
p=0
for(i in 1:length(Newcastle_random)){
if(Newcastle_random[i]=="d"){p=p+1}
  if(Newcastle_random[i]=="w"){p=p+3}
}

Pts_K[13]<-p

Norwich_random <- predict(knnFit, newdata = Norwich)
p=0
for(i in 1:length(Norwich_random)){
if(Norwich_random[i]=="d"){p=p+1}
  if(Norwich_random[i]=="w"){p=p+3}
}

Pts_K[14]<-p

Sheffield_random <- predict(knnFit, newdata = Sheffield)
p=0
for(i in 1:length(Sheffield_random)){
if(Sheffield_random[i]=="d"){p=p+1}
  if(Sheffield_random[i]=="w"){p=p+3}
}

Pts_K[15]<-p

Southampton_random <- predict(knnFit, newdata = Southampton)
p=0
for(i in 1:length(Southampton_random)){
if(Southampton_random[i]=="d"){p=p+1}
  if(Southampton_random[i]=="w"){p=p+3}
}

Pts_K[16]<-p

Tottenham_random <- predict(knnFit, newdata = Tottenham)
p=0
for(i in 1:length(Tottenham_random)){
if(Tottenham_random[i]=="d"){p=p+1}
  if(Tottenham_random[i]=="w"){p=p+3}
}

Pts_K[17]<-p

Watford_random <- predict(knnFit, newdata = Watford)
p=0
for(i in 1:length(Watford_random)){
if(Watford_random[i]=="d"){p=p+1}
  if(Watford_random[i]=="w"){p=p+3}
}

Pts_K[18]<-p

West_random <- predict(knnFit, newdata = West)
p=0
for(i in 1:length(West_random)){
if(West_random[i]=="d"){p=p+1}
  if(West_random[i]=="w"){p=p+3}
}

Pts_K[19]<-p

Wolverhampton_random <- predict(knnFit, newdata = Wolverhampton)
p=0
for(i in 1:length(Wolverhampton_random)){
if(Wolverhampton_random[i]=="d"){p=p+1}
  if(Wolverhampton_random[i]=="w"){p=p+3}
}

Pts_K[20]<-p

```

```{r}
team<-c("Arsenal", "Aston Villa","Bournemouth","Brighton","Burnley","Chelsea","Crystal Palace","Everton","Leicester","Liverpool","Manchester City","Manchester United","Newcastle United","Norwich","Sheffield United","Southampton","Tottenham","Watford","West Ham","Wolverhampton Wanderers")

```




```{r}

Pts_Random=vector()
Arsenal_random <- predict(randomF, newdata = Arsenal)
p=0
for(i in 1:length(Arsenal_random)){
if(Arsenal_random[i]=="d"){p=p+1}
  if(Arsenal_random[i]=="w"){p=p+3}
}

Pts_Random[1]<-p

AS_random <- predict(randomF, newdata = AS)
p=0
for(i in 1:length(AS_random)){
if(AS_random[i]=="d"){p=p+1}
  if(AS_random[i]=="w"){p=p+3}
}

Pts_Random[2]<-p

Bournemouth_random <- predict(randomF, newdata = Bournemouth)
p=0
for(i in 1:length(Bournemouth_random)){
if(Bournemouth_random[i]=="d"){p=p+1}
  if(Bournemouth_random[i]=="w"){p=p+3}
}

Pts_Random[3]<-p


Brighton_random <- predict(randomF, newdata = Brighton)
p=0
for(i in 1:length(Brighton_random)){
if(Brighton_random[i]=="d"){p=p+1}
  if(Brighton_random[i]=="w"){p=p+3}
}

Pts_Random[4]<-p

Burnley_random <- predict(randomF, newdata = Burnley)
p=0
for(i in 1:length(Burnley_random)){
if(Burnley_random[i]=="d"){p=p+1}
  if(Burnley_random[i]=="w"){p=p+3}
}

Pts_Random[5]<-p

Chelsea_random <- predict(randomF, newdata = Chelsea)
p=0
for(i in 1:length(Chelsea_random)){
if(Chelsea_random[i]=="d"){p=p+1}
  if(Chelsea_random[i]=="w"){p=p+3}
}

Pts_Random[6]<-p

Crystal_random <- predict(randomF, newdata = Crystal)
p=0
for(i in 1:length(Crystal_random)){
if(Crystal_random[i]=="d"){p=p+1}
  if(Crystal_random[i]=="w"){p=p+3}
}

Pts_Random[7]<-p

Everton_random <- predict(randomF, newdata = Everton)
p=0
for(i in 1:length(Everton_random)){
if(Everton_random[i]=="d"){p=p+1}
  if(Everton_random[i]=="w"){p=p+3}
}

Pts_Random[8]<-p

Leicester_random <- predict(randomF, newdata = Leicester)
p=0
for(i in 1:length(Leicester_random)){
if(Leicester_random[i]=="d"){p=p+1}
  if(Leicester_random[i]=="w"){p=p+3}
}

Pts_Random[9]<-p

Liverpool_random <- predict(randomF, newdata = Liverpool)
p=0
for(i in 1:length(Liverpool_random)){
if(Liverpool_random[i]=="d"){p=p+1}
  if(Liverpool_random[i]=="w"){p=p+3}
}

Pts_Random[10]<-p

City_random <- predict(randomF, newdata = City)
p=0
for(i in 1:length(City_random)){
if(City_random[i]=="d"){p=p+1}
  if(City_random[i]=="w"){p=p+3}
}

Pts_Random[11]<-p

Manchester_random <- predict(randomF, newdata = Manchester)
p=0
for(i in 1:length(Manchester_random)){
if(Manchester_random[i]=="d"){p=p+1}
  if(Manchester_random[i]=="w"){p=p+3}
}

Pts_Random[12]<-p

Newcastle_random <- predict(randomF, newdata = Newcastle)
p=0
for(i in 1:length(Newcastle_random)){
if(Newcastle_random[i]=="d"){p=p+1}
  if(Newcastle_random[i]=="w"){p=p+3}
}

Pts_Random[13]<-p

Norwich_random <- predict(randomF, newdata = Norwich)
p=0
for(i in 1:length(Norwich_random)){
if(Norwich_random[i]=="d"){p=p+1}
  if(Norwich_random[i]=="w"){p=p+3}
}

Pts_Random[14]<-p

Sheffield_random <- predict(randomF, newdata = Sheffield)
p=0
for(i in 1:length(Sheffield_random)){
if(Sheffield_random[i]=="d"){p=p+1}
  if(Sheffield_random[i]=="w"){p=p+3}
}

Pts_Random[15]<-p

Southampton_random <- predict(randomF, newdata = Southampton)
p=0
for(i in 1:length(Southampton_random)){
if(Southampton_random[i]=="d"){p=p+1}
  if(Southampton_random[i]=="w"){p=p+3}
}

Pts_Random[16]<-p

Tottenham_random <- predict(randomF, newdata = Tottenham)
p=0
for(i in 1:length(Tottenham_random)){
if(Tottenham_random[i]=="d"){p=p+1}
  if(Tottenham_random[i]=="w"){p=p+3}
}

Pts_Random[17]<-p

Watford_random <- predict(randomF, newdata = Watford)
p=0
for(i in 1:length(Watford_random)){
if(Watford_random[i]=="d"){p=p+1}
  if(Watford_random[i]=="w"){p=p+3}
}

Pts_Random[18]<-p

West_random <- predict(randomF, newdata = West)
p=0
for(i in 1:length(West_random)){
if(West_random[i]=="d"){p=p+1}
  if(West_random[i]=="w"){p=p+3}
}

Pts_Random[19]<-p

Wolverhampton_random <- predict(randomF, newdata = Wolverhampton)
p=0
for(i in 1:length(Wolverhampton_random)){
if(Wolverhampton_random[i]=="d"){p=p+1}
  if(Wolverhampton_random[i]=="w"){p=p+3}
}

Pts_Random[20]<-p


```


### Prediction of EPL 2019 using Linear Regression

```{r}
linear_predict<-our_prediction %>%
  mutate(Pts_linear=as.integer(our_prediction$predict_pts_lm),
         Rank=as.integer(prem_data_2019$position),
         predict_rank=as.integer(rank(desc(Pts_linear))),
         diff_rank=position- as.integer(predict_rank),
         diff_pts=pts- as.integer(predict_pts_lm))%>% 
  select(team,pts,Pts_linear,Rank,predict_rank,diff_rank,diff_pts)

linear_predict_edit<-linear_predict %>%
  kbl(caption= "Prediction of EPL 2019 using Linear Regression") %>%
  kable_styling()%>% column_spec(7, color = "white",
              background = spec_color(linear_predict$diff_pts[1:20], end = 0.7)) %>%
  column_spec(6, color = spec_color(linear_predict$diff_rank[1:20]))


linear_predict_edit

```

### Prediction of EPL 2019 using kNN algorithm when k=27

```{r}
Knn_predict<-data.frame(team,Pts_K) 

Knn_predict<-Knn_predict%>% arrange(desc(as.integer(Pts_K))) %>% 
  mutate(Pts=prem_data_2019$pts,
         Rank=as.integer(prem_data_2019$position),
         predict_rank=as.integer(rank(desc(Pts_K))),
         diff_rank=Rank- as.integer(predict_rank),
         diff_pts=Pts- as.integer(Pts_K))

Knn_predict_edit<-Knn_predict %>%
  kbl(caption= "Prediction of EPL 2019 using kNN algorithm when k=27") %>%
  kable_styling()%>% column_spec(7, color = "white",
              background = spec_color(Knn_predict$diff_pts[1:20], end = 0.7)) %>%
  column_spec(6, color = spec_color(Knn_predict$diff_rank[1:20]))


Knn_predict_edit

```

### Prediction of EPL 2019 using Random Forest algorithm

```{r}
Random_predict<-data.frame(team,Pts_Random) 

Random_predict<-Random_predict%>% arrange(desc(as.integer(Pts_Random))) %>% 
  mutate(Pts=prem_data_2019$pts,
         Rank=as.integer(prem_data_2019$position),
         predict_rank=as.integer(rank(desc(Pts_Random))),
         diff_rank=Rank- as.integer(predict_rank),
         diff_pts=Pts- as.integer(Pts_Random))

Random_predict_edit<- Random_predict %>%
  kbl(caption= "Prediction of EPL 2019 using Random Forest algorithm") %>%
  kable_styling()%>% column_spec(7, color = "white",
              background = spec_color(Random_predict$diff_pts[1:20], end = 0.7)) %>%
  column_spec(6, color = spec_color(Random_predict$diff_rank[1:20]))

Random_predict_edit


```

>What we found was that we could predict the rankings of the 2019 season with our linear model, kNN, and random forest. Our linear model prediction of pts was close to the actual pts but prediction of rank predict it as well as we would've hoped. kNN was an even worse prediction than linear model which we can found huge difference between predition and actual. We could found the kNN method limited in understanding the relationship between features and classes. What we were happy with was our random forest prediction. We were very close to the actual rankings with this strategy and many of the teams we correctly predicted.


# Conclusion

>We concluded that through our linear regression model that Expected Goals for and against were significant, which means that it has a positive effect on ranking higher in the Premier football league using the data from 2014 to 2019. Also, we have used linear regression, kNN and Random Forest to predict the six biggest football leagues in six years. We found that kNN is less accurate the more variables there are. Since we had 19 variables in our data, we found a less accurate prediction. Linear regression shows great performance in predicting in pts but not good as much in predict rank. Random Forest was slow in computing but very accurate compared to the kNN model because it is tree based.The result is we were able to predict most of the ranking in the leaderboard, with a few errors that cannot be avoided since there are 29 variables. 

>The limitation of this project is that we only use three method without hyperparameter tuning. We also are working with stats that are calculated after the game which can be difficult to worry about during a match. Moreover, the linear regression model could also be improved by reducing the variables that are not as important. In the future, to improve the model we can also use some better data source since the calculations of expected goals can be confusing and unclear in some of the data.


# Shiny app

runGitHub("eplShiny", user = "whaznaw", ref = "main")

https://github.com/whaznaw/eplShiny
