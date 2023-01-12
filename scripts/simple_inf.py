from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

experiment = 'fcn_hr18s_512x1024_160k_nuimages'
config_file = f'work_dirs/{experiment}/{experiment}.py'
checkpoint_file = f'work_dirs/{experiment}/iter_160000.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
# model.show_result(img, result, out_file=f'pretrained_test_overlay.jpg', opacity=0.5)
model.show_result(img, result, out_file=f'{experiment}_overlay.jpg', opacity=0.5)

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, wait_time=1)
