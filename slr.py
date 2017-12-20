from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use("fivethirtyeight")


def best_fit_slope_and_intercept(xs,ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (mean(xs) ** 2 - mean(xs ** 2))
    b=mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig-ys_line)**2)

def coeff_of_det(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_y_cap = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_y_cap/squared_error_y_mean)

#Generating Training sets
def generate_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step

    xs =[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys,dtype=np.float64)


#Less variance more accurate
xs, ys = generate_dataset(100, 20, 2, correlation='pos')

#Slope and Intercept calculated based on training data set
m, b = best_fit_slope_and_intercept(xs,ys)

#Predicted y values for training set x values
regression_line = [m*x+b for x in xs]

#Test values
test_x = 63.05
test_y = m*test_x + b

#Graph of original x, y values
plt.scatter(xs,ys)

#Graph of original x and predicted y values
plt.plot(xs,regression_line)

#Graph for test value
plt.scatter(test_x,test_y,color="red")
plt.title("Linear Regression").set_size(15)

print("Predicted y value",test_y)

plt.show()

#We can predict y value for a new x
#newx = 20
#predicted_y = m*newx+b
#print(predicted_y)

#Shows accuracy. Higher value of r_squared shows more accuracy
r_squared = coeff_of_det(ys, regression_line)
print("Accuracy",r_squared)