import os
import cv2

def process_all_trian_data(input_path1,input_path2,output_path):
    '''
    处理所有训练图片

    Args:
        input_path1: the file_path where store all canny edge pictures
        input_path2: the file_path where store all HED edge pictures
        output_path: the train_data path
    
    Author: Zhongqi Wang
    '''

    files_path1 = []
    files_path2 = []

    for label in sorted(os.listdir(input_path1)): #label：来源哪个数据集
        for fname in os.listdir(os.path.join(input_path1, label)):
            files_path1.append(os.path.join(input_path1, label, fname)) #图片的文件名

    for label in sorted(os.listdir(input_path2)): #label：来源哪个数据集
        for fname in os.listdir(os.path.join(input_path2, label)):
            files_path2.append(os.path.join(input_path2, label, fname)) #图片的文件名

    assert(len(files_path1)==len(files_path2))

    for i in range(len(files_path1)):
        file1 = files_path1[i]
        file2 = files_path2[i]

        name1 = file1.split("\\")
        label = name1[-2] #获取数据集名，如met-1
        fname = name1[-1] #获取文件名，如met_0.jpg
        arguments_strOut = os.path.join(output_path, label, fname)

        pic1 = cv2.imread(file1)
        pic1 = cv2.resize(pic1, (512, 512))
        pic2 = cv2.imread(file2)

        train_data = pic1+pic2

        cv2.imwrite(arguments_strOut,train_data)


if __name__ == "__main__":
    input_path1 = "canny_pic" 
    input_path2 = "processing_pic" 
    output_path = "train_data" 
    process_all_trian_data(input_path1,input_path2,output_path)