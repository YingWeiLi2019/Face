import sys
import os
from math import pow
from PIL import Image,ImageDraw,ImageFont
import cv2
import math
import random
caffe_root = '/home/matt/Document/caffw/'

sys.path.insert(0,caffe_root + 'python')
os.environ['GLOG_minloglevel'] = '2'
import caffe

#1.进行图片预处理
def face_dection(imgFile):
	net_full_conv = caffe.Net('deploy_full_conv.prototxt',
								'caffemodel',
								caffe.Test)
	scales = []
	factor = 0.79
	img = cv2.imread(imgFile)
	largest = largest = min(2,4000/max(img.shape[0:2]))
	largest = largest
	minD = largest*min(img.shape[0:2])
	while minD >= 227:
		scales.append(scale)
		scale *= factor
		minD *= factor

	total_boc = []
	for scale in scales:
		scale_img = cv2.resize(img,((int(img.shape[0]*scale)),int(img.shape[1]*scale)))
		cv2.imwrte('path',scale_img)
		im = caffe.io.load_image('path')
		net_full_conv.blobs['data'].reshape(1,3,scale_img[0])
		transformer = caffe.io.Transformer({'data':net_full_blobs['data'].data.shape})
		transformer.set_mean('data',np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
		transformer.set_transpose('data',(2,0,1))
		transformer.set_channel_swap('data',(2,1,0))
		transformer.set_raw_scale('date',255)
#2.前向传播
		out = net_full_conv.forward_all(data = np.asarray(transformer.preprocess('data',im)))
#3.非极大值抑制算法NMS对人脸检测框的最优选择
		boxes = generateBoundingBox(out['prob'][0,1],scale)
		if (boxes):
			total_box.append(boxes)
	boxes_nms = np.array(total_box)
	ture_boxes = nms(boxes_nms,0.8)
#4.如果box检测完了，那么就要对box进行画出来，这时可以使用opendcv工具进行截取
	if not ture_boxes == []:
		x1,y1,x2,y2 = ture_boxes
		cv2.rectangle(img,(int(x1),int(y1),int(x2),int(y2)))
		cv2.imshow()
#对样本进行判断，是否为正样本为人脸
def generateBoundingBox(featureMap,scale):
	boundingBox = []
	stride = 32
	cellSize = 227
	for (x,y),prob in np.ndenumerate(featureMap):
		#在这里判断概率值大于0.95则是人脸
		if prob>0.95:
			#如果是人脸人就把它放到Box里边，但是在放的时候要进行处理一些坐标的反变换
			boundingBox.append(float(stride*y)/scale,float(stride*x)/scale,
								(float(stride*y)/scale+cellSize)/scale,(float(stride*x)/scale+cellSize)/scale
								,prob)
			return boundingBox
#非极大值抑制算法NMS
#表示：在人脸检测的时候，会出现很多检测框，但是我们只需要一个框就足够了，所以要进行筛选
#怎么筛选?可以根据IOU来筛选，如果两个框的IOU>0.8，那么它们是高度重叠的
#则要分别对它们进行概率的计算，然后进行比较，哪个概率最小就去掉哪个，然后选择最大的那个。
#重复上面的步骤，最后选出最大概率的框。则是我们想要的