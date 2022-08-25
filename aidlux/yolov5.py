# aidlux相关
from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res

# 七牛云相关
from qiniu import Auth, put_file
from qiniu import CdnManager
import time
import requests
import cv2

# 配置七牛云信息
access_key = ""   # 此处需替换成自己的
secret_key = ""   # 此处需替换成自己的
bucket_name = "aidlux202208"
bucket_url = "rgu20nh7f.hn-bkt.clouddn.com"
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)


# 将本地图片上传到七牛云中
def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name)
    put_file(token, file_name, file_path)


# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url


# 加载模型
model_path = 'best-fp16.tflite'
# 定义输入输出shape
in_shape = [1 * 640 * 640 * 3 * 4]
out_shape = [1 * 25200 * 6 * 4, 1 * 3 * 80 * 80 * 6 * 4, 1 * 3 * 40 * 40 * 6 * 4, 1 * 3 * 20 * 20 * 6 * 4]

# 载入模型
aidlite = aidlite_gpu.aidlite()
# 载入yolov5检测模型
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)

cap = cvs.VideoCapture(0)
fire = 0
while True:
    frame = cap.read()
    if frame is None:
        continue

    # 预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)

    aidlite.setInput_Float32(img, 640, 640)
    # 推理
    aidlite.invoke()
    pred = aidlite.getOutput_Float32(0)
    pred = pred.reshape(1, 25200, 6)[0]
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.5, iou_thres=0.45)
    res_img, fire = draw_detect_res(frame, pred, fire)

    if fire == 1:
        cv2.imwrite("detect_image.jpg", res_img)
        fire = 0
        # 需要上传到七牛云上面的图片的路径
        image_up_name = "detect_image.jpg"
        # 上传到七牛云后，保存成的图片名称
        image_qiniu_name = "detect_image_2023.jpg"
        # 将图片上传到七牛云,并保存成image_qiniu_name的名称
        upload_img(bucket_name, image_qiniu_name, image_up_name)
        # 取出和image_qiniu_name一样名称图片的url
        url_receive = get_img_url(bucket_url, image_qiniu_name)
        print(url_receive)
        # 需要刷新的文件链接,由于不同时间段上传的图片有缓存，因此需要CDN清除缓存，
        urls = [url_receive]
        # URL刷新缓存链接,一天有500次的刷新缓存机会
        refresh_url_result = cdn_manager.refresh_urls(urls)

        # 填写对应的喵码
        id = 'ti5evnL'
        # 填写喵提醒中，发送的消息，这里放上前面提到的图片外链
        text = "告警图片：" + url_receive
        ts = str(time.time())  # 时间戳
        type = 'json'  # 返回内容格式
        request_url = "http://miaotixing.com/trigger?"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

        result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=" + type,
                               headers=headers)
    cvs.imshow(res_img)