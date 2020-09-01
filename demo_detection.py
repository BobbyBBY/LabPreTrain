import cv2
 
vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\1598669367563.MP4")  # 读入视频文件
# vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\新视频.MP4")  # 读入视频文件
# vc = cv2.VideoCapture("C:\\Users\\76449\\Videos\\Captures\\gitrevert.MP4")  # 读入视频文件
# vc = cv2.VideoCapture(0)  # 打开摄像头
 
rval, firstFrame = vc.read()
firstFrame = cv2.resize(firstFrame, (640, 360), interpolation=cv2.INTER_LINEAR)# interpolation：插值方法
gray_firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)   # 灰度化
firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)      #高斯模糊，用于去噪
prveFrame = firstFrame.copy()
 
#遍历视频的每一帧
while True:
    # ret表示是否正确读取到了帧
    # frame 每一帧，三维矩阵
    (ret, frame) = vc.read()
 
    # 如果没有获取到数据，则结束循环。如视频结束
    if not ret:
        break
 
    # 对获取到的数据进行预处理
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("oriegin_frame", frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_frame", gray_frame)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    cv2.imshow("gaussian_blur", gray_frame)
    cv2.imshow("prveFrame", prveFrame)
 
    # 计算当前帧与上一帧的差别
    frameDiff = cv2.absdiff(prveFrame, gray_frame)
    cv2.imshow("frameDiff", frameDiff)
    prveFrame = gray_frame.copy()
 
    # 图像的二值化
    # 忽略较小的差别
    retVal, thresh = cv2.threshold(frameDiff, 20, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh1', thresh)
 
    # 对阈值图像进行填充补洞
    thresh = cv2.dilate(thresh, None, iterations=20)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    text = "Not Moving"
    # 遍历轮廓
    for contour in contours:
        # if contour is too small, just ignore it
        if cv2.contourArea(contour) < 500:   #面积阈值
            continue
 
        # 计算最小外接矩形（非旋转）
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving"
 
    cv2.putText(frame, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    cv2.imshow('frame_with_result', frame)
    cv2.imshow('thresh2', thresh)
    cv2.imshow('frameDiff', frameDiff)
 
    # 处理按键效果
    key = cv2.waitKey(60) & 0xff
    if key == 27:  # 按下ESC时，退出
        break
    elif key == ord(' '):  # 按下空格键时，暂停
        cv2.waitKey(0)
 
    cv2.waitKey(0)
 
vc.release()