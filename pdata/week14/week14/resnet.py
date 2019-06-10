from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image
import numpy as np

#导入预训练模型ResNet50
model = ResNet50(weights='imagenet')

#对输入图片进行处理
img_path =  'c:/pdata/week14/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
X = image.img_to_array(img) #将图像转换为数组
X = np.expand_dims(X, axis=0)
X = preprocess_input(X)  

#模型预测
preds = model.predict(X)
print('Predicted:', decode_predictions(preds, top=3)[0])
