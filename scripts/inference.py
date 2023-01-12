#!/usr/bin/env python
# license removed for brevity

import rospy
from sensor_msgs.msg import Image
import PIL.Image

from mmseg.apis import inference_segmentor, init_segmentor

PALETTE = [
        [1, 1, 1],  # unknown
        [245, 254, 184],  # driveable surface
        [95, 235, 52],  # humans
        [52, 107, 235],  # moveable object
        [150, 68, 5],  # vehicles
    ]

config_file = 'fcn_hr18_512x1024_160k_nuimages.py'
checkpoint_file = 'iter_160000.pth'

# build the model from a config file and a checkpoint file
MODEL = init_segmentor(config_file, checkpoint_file, device='cuda:0')
# # test a single image and show the results


def image_callback(image_data):
    pub = rospy.Publisher('semantic_mask', Image, queue_size=10)

    # getting the result from the model
    seg_mask = inference_segmentor(MODEL, image_data)

    # converting to coloured masks
    seg_img = PIL.Image.fromarray(seg_mask).convert("P")
    seg_img.putpalette(PALETTE)

    # publish masks
    pub.publish(seg_img)

    return None


def main():

    rospy.init_node('semantic_masks', anonymous=True)
    rospy.Subscriber('/cam_front/raw', Image, image_callback)

    rospy.spin()


if __name__ == '__main__':
    # try:
    #     talker()
    # except rospy.ROSInterruptException:
    main()
