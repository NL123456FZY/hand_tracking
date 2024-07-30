import cv2
import mediapipe as mp

print("Starting...")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()