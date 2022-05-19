set -x
# #IFS=$'\n'
# #for model in `cat models.txt`
# #do
# #echo $model
# model="EfficientNetV2"
# #lr="0.00001"
# #python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/train.ECAD.CH1 -v /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1 -m $model  -o Nadam -p 256 -l /home/napr4836/results/test_image_regression_4_29_2022  -r $lr -L log_cosh -e 200 -b 64
# lr="0.000001"
# #python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/train.ECAD.CH1 -v /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1 -m $model  -o Nadam -p 256 -l /home/napr4836/results/test_image_regression_4_29_2022  -r $lr -L log_cosh -e 200 -b 64
# loss="MeanAbsoluteError"
# lr="0.00001"
# #python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/train.ECAD.CH1 -v /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1 -m $model  -o Nadam -p 256 -l /home/napr4836/results/test_image_regression_4_29_2022  -r $lr -L $loss -e 200 -b 64
# loss="MeanSquaredError"
# #lr="0.00001"
# lr="0.0001"
# #python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/train.ECAD.CH1 -v /home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1 -m $model  -o Nadam -p 256 -l /home/napr4836/results/test_image_regression_4_29_2022  -r $lr -L $loss -e 200 -b 64

# #model_path="/home/napr4836/results/test_image_regression_4_29_2022/EfficientNetV2_Nadam_1e-06-log_cosh"
# #model_path="/home/napr4836/results/test_image_regression_4_29_2022/EfficientNetV2_Nadam_1e-05-MeanAbsoluteError"

######################################################

### Runs predictions 

######################################################

#model = "EfficientNetV2"
#model_path="/home/napr4836/results/prep_train_val_test_0.75_train.ECAD.CH1/EfficientNetV2_Nadam_0.0001-MeanSquaredError"
#input_file_path="/home/napr4836/temp/prep_train_val_test_0.75/val.ECAD.CH1"
#output_file="/home/krishna/tttt/val.ECAD.CH1.xls"
#python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file

#######################################################


# input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/test.ECAD.CH1"
# output_file=$model_path"/test.ECAD.CH1.xls"
# python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file
# #done

#how to run
#training
#python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t <Path to training file with image file path and value to predict> -v <path to validation file> -m <model>  -o <optimizer> -p <Image patch size> -l <Log directory> -r <Learning rate> -L <Loss function> -e <Num epochs> -b <Batch size>

#prediction script
#python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  <Path to train/val/test file with image file path and value to predict> <model log directory> <output file>


#################################################

##### Training ########

#################################################

#example runs
#training
loss="MeanSquaredError"
lr="0.0001"
model="EfficientNetV2"
python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train_auto.py -t /home/napr4836/temp/prep_train_val_test_0.75/train.ECAD.CH1 -v /home/napr4836/temp/prep_train_val_test_0.75/val.ECAD.CH1 -m $model  -o Nadam -p 256 -l /home/krishna/tttt/results/  -r $lr -L $loss -e 200 -b 64

##################################################

#prediction val & test
#model_path="/home/napr4836/results/test_image_regression_4_29_2022/EfficientNetV2_Nadam_0.0001-MeanSquaredError"
#input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1"
#output_file=$model_path"/val.ECAD.CH1.xls"
#python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file

#input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/test.ECAD.CH1"
#output_file=$model_path"/test.ECAD.CH1.xls"
#python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file
