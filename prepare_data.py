
import string
import random
import matplotlib.pyplot as plt

from captcha.image import ImageCaptcha # 產生驗證碼的套件


# create 0~9, A~Z
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 128, 64, 4, len(characters)


generator = ImageCaptcha(width=width, height=height)

#str.join => 已指定字符連接字串 ex "-"".join(str)=str1-str2-str3-st34
randomStr = ''.join([random.choice(characters) for j in range(n_len)])
img = generator.generate_image(randomStr)


plt.imshow(img)
plt.show()

