import numpy as np
import cv2


def draw_results_ssd(detected,input_img,faces,ad,img_size,img_w,img_h,time_detection,time_network,time_plot):
    if detected.shape[2]>0:
        for i in range(0, detected.shape[2]):
            confidence = detected[0, 0, i, 2]

            if confidence > 0.5:
                (h0, w0) = input_img.shape[:2]
                box = detected[0, 0, i, 3:7] * np.array([w0, h0, w0, h0])
                (startX, startY, endX, endY) = box.astype("int")
                x1 = startX
                y1 = startY
                w = endX - startX
                h = endY - startY
                
                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                
                faces[i,:,:,:] = cv2.resize(input_img[yw1 : yw2+1, xw1 : xw2+1, :], (img_size, img_size))
                faces[i,:,:,:] = cv2.normalize(faces[i,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
                
                face = np.expand_dims(faces[i,:,:,:], axis=0)
    
    try:
        fd_signal = 1
        return face, fd_signal  
    
    except UnboundLocalError:
        face = np.ones(shape=(1,224,224,3))
        fd_signal = 0
        return face, fd_signal