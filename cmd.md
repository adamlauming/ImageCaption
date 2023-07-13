<!--
 * @Description: 
 * @Author: Ming Liu (lauadam0730@gmail.com)
 * @Date: 2021-06-17 12:48:43
-->
cd /home/su2/ImageCaption/codes

# lambda_stop adjustment
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=8 --lambda_word=1 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop8-Lword1;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=5 --lambda_word=1 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop5-Lword1;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=1 --lambda_word=1 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1;

# 加入 feature平均值(FC feature)作为LSTM初始输入(h0,c0)
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=0 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1-Lcla0;

# 加入 classification loss
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=1 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1-Lcla1;
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=2 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1-Lcla2;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=5 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1-Lcla5;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=6 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=10 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop1-Lword1-Lcla10;

python train.py --savename=Caption1;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=0 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla0;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=2 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla2;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla3;

# co-attention模块
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla3;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption3 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption3-Lstop3-Lword1-Lcla3;

# spatial correlation module
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption4 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption4-SCM;

# PSP Module
python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=8 --model=Caption5 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption5-PSP;

# self-attention module



# batch 4  image size 512
python train.py --savename=Caption1 --batch_size=4 --img_size=512;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=4 --img_size=512 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=0 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla0;python vhrnn_train.py --gpu_rate=0.8 --epochs=50 --workers=4 --batch_size=4 --img_size=512 --model=Caption2 --encoder=resnet50 --lambda_stop=3 --lambda_word=1 --lambda_cla=3 --learning_rate=2e-4 --datatrain=train --savename=Caption2-Lstop3-Lword1-Lcla3;