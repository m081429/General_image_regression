# set -x
# #IFS=$'\n'
# #for model in `cat models.txt`
# #do
# #echo $model
# 
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
# model_path="/home/napr4836/results/test_image_regression_4_29_2022/EfficientNetV2_Nadam_0.0001-MeanSquaredError"
# input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1"
# output_file=$model_path"/val.ECAD.CH1.xls"
# python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file

# input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/test.ECAD.CH1"
# output_file=$model_path"/test.ECAD.CH1.xls"
# python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file
# #done


#example runs
#training
loss="MeanSquaredError"
lr="0.0001"
model="EfficientNetV2"
model_sub_dir="EfficientNetV2_Nadam_0.0001-MeanSquaredError"

opt="Nadam"
log_dir="/home/napr4836/results/test_image_regression_4_29_2022"
num_epoch="200"
batch_size="64"

#input_train="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/train.ECAD.CH1"
#input_val="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1"
#log_dir="/home/napr4836/results/test_image_regression_4_29_2022"
for i in `ls /home/napr4836/temp/prep_train_val_test_0.75/train*`
do
    input_train=$i
    input_val=`echo $input_train|sed -e 's/\/train./\/val./g'`
    log_dir=`echo $input_train|rev|cut -f1-2 -d '/'|rev|sed -e 's/\//_/g'`
    log_dir="/home/napr4836/results/$log_dir"
    echo $input_train $input_val $log_dir
    # echo "python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t $input_train -v $input_val -m $model  -o $opt -p 256 -l $log_dir -r $lr -L $loss -e $num_epoch -b $batch_size" > $log_dir".sh"
    # python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/train.py -t $input_train -v $input_val -m $model  -o $opt -p 256 -l $log_dir -r $lr -L $loss -e $num_epoch -b $batch_size > $log_dir".log" 2>&1
    
    model_path_sub="$log_dir/$model_sub_dir"
    output_file=$model_path_sub"/train.pred.xls"
    python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_train $model_path_sub $output_file
    
    model_path_sub="$log_dir/$model_sub_dir"
    output_file=$model_path_sub"/val.pred.xls"
    python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_val $model_path_sub $output_file
    
    input_test=`echo $input_train|sed -e 's/\/train./\/test./g'`
    model_path_sub="$log_dir/$model_sub_dir"
    output_file=$model_path_sub"/test.pred.xls"
    python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_test $model_path_sub $output_file
    
    #exit
done
# #prediction val & test
# model_path="/home/napr4836/results/test_image_regression_4_29_2022/EfficientNetV2_Nadam_0.0001-MeanSquaredError"
# input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/val.ECAD.CH1"
# output_file=$model_path"/val.ECAD.CH1.xls"
# python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file

# input_file_path="/home/napr4836/scripts/ANHIR_MW/Registration/prep_train_val_test_0/test.ECAD.CH1"
# output_file=$model_path"/test.ECAD.CH1.xls"
# python /home/napr4836/scripts/ANHIR_MW/Registration/Image_regression/batch_predict.py  $input_file_path $model_path $output_file
