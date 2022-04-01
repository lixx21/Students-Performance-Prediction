# Students-Performance-Analysis

The purpose of this project is to analyze student performance in their exams and to understand the correlation between each features 

## Dataset

The dataset is from kaggle, you can click the link below:

[kaggle Student's Performance ](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

## Data Analyze

From this analyze we can know that:

 1. The dataset has 1000 rows and 8 columns
 2. The statistics info of dataset
 3. Number of female students is bigger than male students
 4. The most number from race/ethnicity is from Group C
 5. Parental level education mostly from some college
 6. Most of students have a standard lunch
 7. Most of students didn't take any test preparation course
 8. There is no missing data or duplicated data

The correlation method that used in this case is 'Pearson' which is standard correlation coefficient that measures the strength of the linear relationship between two variables. From this correlation method we know a lot of information about correlation for each attributes / columns. The correlation of each features are: 

![Screenshot 2022-04-01 170550](https://user-images.githubusercontent.com/91602612/161232306-f350d90a-977c-47c1-8a07-425bc5e5cced.png)

# Students-Performance-Prediction

Now after we analyze the students' performance we want to make a prediction for student's exam score using linear regression algorithm. we divide the data with 70% of data train and 30% of data test / validation. 

## Math Exam Prediction
 1. the coefficient of the model are: [-13.12, 0.72, -0.15, -3.1, -3.58, 0.28, 0.68]
 2. the intercept of the model is: 8.56
 3. Prediction Accuracy: 87%

## Reading Exam Prediction
 1. the coefficient of the model are: [0.37, -0.35, 0.06, 1.19, -1.34, 0.15, 0.82]
 2. the intercept of the model is: 3.39
 3. Prediction Accuracy: 92%

## Reading Exam Prediction
 1. the coefficient of the model are: [5.99, 0.13, 0.05, -0.27, 3.31, 0.29, 0.66]
 2. the intercept of the model is: -1/57
 3. Prediction Accuracy: 93%
