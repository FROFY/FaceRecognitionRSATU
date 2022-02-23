import PIL.Image
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw
from IPython import display
from facenet_pytorch import MTCNN, InceptionResnetV1


cam = cv2.VideoCapture(0)


def Main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)

    mtcnn = MTCNN(keep_all=True, device=device)

    while cam.isOpened():
        _, frame = cam.read()
        if _ is False:
            print('Error in camera')
            return
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(frame_gray)
        frame_draw = frame.copy()

        pil_image = PIL.Image.fromarray(frame_draw)
        draw = ImageDraw.Draw(pil_image)

        if boxes is not None:
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(0, 0, 255), width=5)
                draw.text((box.tolist()[3], box.tolist()[3]), 'Human', fill=(255, 255, 255, 255))

        image = np.array(pil_image)
        cv2.imshow('FROFY', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()
