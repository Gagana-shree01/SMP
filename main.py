import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Student_Marks.csv")
print(data.head(10))
# The dataset consists of just three columns. Since we must forecast a student's grades, the marks column is the target column.
data.info()

pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(data)

print(data.isnull().sum()) # checking if null vlues exixt
# Since there are no null values in the dataset, it is suitable for usage. 

data['number_courses'].unique()
data["number_courses"].value_counts() # The data includes a column that lists the number of courses that each student has selected.

# So students have chosen a minimum of three and a maximum of eight courses. Let's look at a scatter plot to determine if a student's course load influences his or her grades.
figure = px.scatter(data_frame=data, x = "number_courses", 
                    y = "Marks", size = "time_study", 
                    title="Number of Courses and Marks Scored")
figure.show()

#According to the data visualisation above, the number of courses a student takes may not effect his or her grades if the student studies for longer hours each day. So, let us examine the link between the amount of time spent studying each day and the grades earned by the student.
figure = px.scatter(data_frame=data, x = "time_study", 
                    y = "Marks", size = "number_courses", 
                    title="Time Spent and Marks Scored", trendline="ols")
figure.show()

correlation = data.corr()
print(correlation["Marks"].sort_values(ascending=False))

x = np.array(data[["time_study", "number_courses"]])
y = np.array(data["Marks"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

#From above graph , there is a linear relationship between the time studied and the marks obtained. This means the more time students spend studying, the better they can score.
model = LinearRegression()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)    # The model gives 94%-95% accuracy

# result presdiction example
prediction = np.array([[4.508, 3]])
model.predict(prediction)
