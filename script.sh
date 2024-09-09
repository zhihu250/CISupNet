#######Our Model
# train_bs ResNet50 for ccvid:
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ccvid --cfg configs/c2dres50_ce_cal.yaml --gpu 0,1
# for prcc
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 #

python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1

# 模型测试：
##在configs.default_img.py 中选择# _C.INFER_MODE = True 
#添加模型权重路径：_C.MODEL.RESUME = '/home/ta/gcx/Simple-CCReID-main/data/logs/ltcc/res50-cels-cal/best_model.pth.tar'