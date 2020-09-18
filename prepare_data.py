import yaml
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from captcha.image import ImageCaptcha # 產生驗證碼的套件

data_config_path = "./conf/data.yaml"

class generatorData:

    def __init__(self):
        
        self.config = self.get_Config()
        self.characters = string.digits + string.ascii_uppercase
        self.width = self.config["data"]["Captcha"]["width"]
        self.height = self.config["data"]["Captcha"]["height"]
        self.word = self.config["data"]["Captcha"]["wordnum"]
        self.classNum = len(self.characters)
        


    def get_Config(self):
        with open(data_config_path, "r") as stream:
            data = yaml.load(stream, Loader=yaml.FullLoader)
        return data
    
    def gen_Data(self, batch_size=8):
        x = np.zeros((batch_size, self.height, self.width, 3), dtype= np.uint8)
        y = [np.zeros((batch_size, self.classNum), dtype=np.uint8) for i in range(self.word)]
        print(np.shape(y))
        # ImageCaptcha instance
        generator = ImageCaptcha(width=self.width, height=self.height)
        while True:
            for i in range(batch_size):
                random_captcha = ''.join([random.choice(self.characters) for j in range(4)])
                x[i] = generator.generate_image(random_captcha)
                for j, ch in enumerate(random_captcha):
                    y[j][i, :] = 0
                    y[j][i, self.characters.find(ch)] = 1
            data_list = []
            for single in range(batch_size):
                for data in y:
                    w = np.argmax(data[single, :]) + 1
                    data_list.append(w)
            yield x, data_list ## x == input images, data_list=list of label 


gendata = generatorData()
loader = gendata.gen_Data()

print(next(loader)[1])

"""
# create 0~9, A~Z
characters = string.digits + string.ascii_uppercase

width, height, n_len, n_class = 128, 64, 4, len(characters)


generator = ImageCaptcha(width=width, height=height)

#str.join => 已指定字符連接字串 ex "-"".join(str)=str1-str2-str3-st34
randomStr = ''.join([random.choice(characters) for j in range(n_len)])
img = generator.generate_image(randomStr)



#plt.imshow(img)
#plt.show()

n = 1 ## 給CTC的<NULL>類別使用 
character_index = {} ## character對照標籤 dict
index_character = {} ## 標籤對照character dict
for character in characters:
    character_index[character] = n
    index_character[n] = character
    n += 1

print(character_index)
print(index_character)

def gen(batch_size=8):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_code = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_code)
            for j, ch in enumerate(random_code):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        data_list = []
        for single in range(batch_size):
            for data in y:
                w = np.argmax(data[single, :]) + 1
                data_list.append(w)
        yield X, data_list # X == input images, data_list=list of label 


loader = gen(batch_size=8)

print(type(loader))

print(next(loader)[0].shape)
print(next(loader)[1])


X_train , target = next(loader)


print(X_train[1].shape)
print(target)
"""