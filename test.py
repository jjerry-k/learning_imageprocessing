import cv2

a = cv2.imread('./lena.jpg')
print(a.shape)
cv2.imshow('image', a)
cv2.waitKey(0)
cv2.destroyAllWindows()
