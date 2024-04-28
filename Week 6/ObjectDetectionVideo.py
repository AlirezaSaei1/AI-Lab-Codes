import cv2

# Replace with the video path
video_path = 'F:\\University 8 (Final)\\AI Lab\\AI-Lab-Codes\\Week 6\\Media\\Ball_Bounce.mp4'
video = cv2.VideoCapture(video_path)


def detect_red_ball(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    cr_channel = ycrcb[:,:,1]

    _, red_mask = cv2.threshold(cr_channel, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'w={w}, h={h}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


while True:
    ret, frame = video.read()

    if not ret:
        break

    frame_with_red_balls = detect_red_ball(frame)

    cv2.imshow('Ball Detection', frame_with_red_balls)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()