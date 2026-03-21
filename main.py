import cv2 
import numpy as np
from utils import detectar_pcb, redimensionar_imagem, comparar_pcb


img = cv2.imread("esp32.jpg")
img2 = cv2.imread("esp32-2.jpg")
img3 = cv2.imread("esp32-3.jpg")

img = redimensionar_imagem(img)
img2 = redimensionar_imagem(img2)
img3 = redimensionar_imagem(img3)

img = detectar_pcb("",img)
img2 = detectar_pcb("",img2)
img3 = detectar_pcb("",img3)



test = comparar_pcb(img, img2, 50)
cv2.imshow("testando 1 e 2", test)

test = comparar_pcb(img, img3, 50)
cv2.imshow("testando 1 e 3", test)

# redimensionar_imagem(img, )
# red = cv2.imread("redimensionada.jpg")
# cv2.imshow("teste def redimensionar", red)


# teste = detectar_pcb("redimensionada.jpg")
# print(teste)

# # cv2.imshow("th", thresh)


# # cv2.imshow("img",nova_img)
# cv2.imshow("roi teste",teste)
cv2.waitKey(0)
cv2.destroyAllWindows()