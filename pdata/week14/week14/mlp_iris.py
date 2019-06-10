import pandas as pd
filename = 'c:/pdata/week14/iris.data'
data = pd.read_csv(filename, header = None)
data.columns = ['sepal length','sepal width','petal length','petal width','class']
X = data.iloc[:,0:4].values.astype(float)
y = pd.get_dummies( data[‘class’], prefix=‘class’ )  #多分类问题，y要转化为三列二元标签
y = y.values.astype(int)

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)

from keras.models import Sequential
from keras.layers import Dense, Activation

# 定义模型结构
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(4,)))
model.add(Dense(16, activation="relu"))
model.add(Dense(3, activation="softmax"))
#定义损失函数和优化器，并编译
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2, validation_data=(X_test,y_test))
# 评估模型
loss, accuracy = model.evaluate(X_test,y_test, verbose=2)
print('loss = {},accuracy = {} '.format(loss,accuracy) )
classes = model.predict(X_test, batch_size=1, verbose=2)
print('测试样本数：',len(classes))
print("分类概率:\n",classes)
