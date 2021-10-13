import cv2 as cv
import mediapipe as mp

def detection(image):
	img=cv.cvtColor(image, cv.COLOR_BGR2RGB)

	mp_hand=mp.solutions.hands
	hand=mp_hand.Hands()
	mpDraw=mp.solutions.drawing_utils
	result=hand.process(img)
	hand_landmarks=result.multi_hand_landmarks

	fingerCoordinates=[(8,6),(12,10),(16,14),(20,18)]
	thumCoordinates=(4,2)

	if hand_landmarks:
		handPoints=[]
		for handLms in hand_landmarks:
			mpDraw.draw_landmarks(img, handLms,mp_hand.HAND_CONNECTIONS)
			for idx,lm in enumerate(handLms.landmark):
				h,w,c=img.shape
				cx,cy=int(lm.x*w),int(lm.y*h)
				handPoints.append((cx,cy))
	upCount=0

	for cordinate in fingerCoordinates:
		if handPoints[cordinate[0]][1] < handPoints[cordinate[1]][1]:
			upCount+=1

	if handPoints[thumCoordinates[0]][0] > handPoints[thumCoordinates[1]][0]:
		upCount+=1
	
	cv.putText(img, str(upCount), (10,10), cv.FONT_ITALIC, 5 , (250,0,0),2)		

	return img


img=cv.imread('one.jpg')
cv.imshow('f', detection(img))

cv.waitKey(0)
cv.destroyAllWindows()   