import cv2, os, random, numpy, pandas

# Change the path accordingly
path = './CompCars_Dataset/data/data/image/'


classes = []
data = pandas.DataFrame(columns=["path_to_image", "class", "class_index"])
for make in os.listdir(path):
    make_path = path + make
    for model in os.listdir(make_path):
        model_path = make_path + '/' + model
        for year in os.listdir(model_path):
            class_name = str(model) + '_' + str(year)
            if class_name not in classes:
                classes.append(class_name)
            year_path = model_path + '/' + year
            for image in os.listdir(year_path):
                im = year_path + '/' + image
                data = data.append({"path_to_image": im, "class": str(model) + '_' + str(year), "class_index":
                    classes.index(str(model) + '_' + str(year))}, ignore_index=True)

num_classes = len(classes)
print(num_classes)
data = data.sample(frac=1)
data_train = data.iloc[:int(len(data.index) * 0.8)]
data_test = data.iloc[int(len(data.index) * 0.8):]
data_train.to_csv('new_train.csv')
data_test.to_csv('new_test.csv')

