import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pydotplus

# Savings the values form the CVS files
df = pandas.read_csv("train.csv");
test_data = pandas.read_csv("test.csv")

# Converting String variables from Sex into ints
d = {'female': 0, 'male': 1}

# Replacing each female with 0 and each male with 1
df['Sex'] = df['Sex'].map(d)
test_data['Sex'] = test_data['Sex'].map(d)

# Converting string embarked location to an int for comparison
d = {'S': 0, 'C': 1, 'Q': 2}

# Replacing each embarking string with relevant int
# Using the mean value can result in a bias being formed
#df['Embarked'] = df['Embarked'].map(d)
#test_data['Embarked'] = test_data['Embarked'].map(d)

# # Cleaning up missing embarking
#df['Embarked'].fillna(df['Embarked'].median(), inplace=True)
#test_data['Embarked'].fillna(test_data['Embarked'].median(), inplace=True)

df = df.dropna()
test_data = test_data.dropna()


# Clearing up missing Fares
# df['Fare'].fillna(df['Fare'].mean(), inplace=True)
# test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# Removing NaN fields in Age and replacing with the mean
# A bias is created here. Better method at guessing missing ages
# Will result in more accurate results
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Generates a new social economic field using age and class
df['Age*Class'] = df['Age'] * df['Pclass']
test_data['Age*Class'] = test_data['Age'] * test_data['Pclass']

# Generates family size field
df['FamilySize'] = df['SibSp'] + df['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Stores the fields that will be compared
features = ['Pclass', 'Sex', 'Age', 'Parch', 'SibSp', 'Age*Class', 'Embarked']
X = df[features]
X_Test = test_data[features]

# Sets the test condtions 1 to survive, 0 to die
y = df['Survived']

# Uses a RandomForsestClassifier to plot train date, then makes predictions using test data
RFC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
RFC.fit(X, y)
RFCPredictions = RFC.predict(X_Test)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
predictions = dtree.predict(X_Test)

output = pandas.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

output = pandas.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': RFCPredictions})
output.to_csv('RFCSubmission.csv', index=False)

print("Your submission was successfully saved!")

data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.jpg')
img = pltimg.imread('mydecisiontree.jpg')
imgplot = plt.imshow(img)
plt.show()
