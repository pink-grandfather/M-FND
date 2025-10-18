## Base data folder

######## MediaEval ######## 

## Train and Test csvs
root_dir = 'E:/GAME-ON-main/creating-graph-data/dataset/twitter/'
me15_train_csv_name = 'train.csv'
me15_test_csv_name = 'test.csv'

me15_image_vec_dir = "features/"

me15_text_vec_dir = "bert_tokens/"


######## Weibo ########

## Train and Test csvs
we_train_csv_name = "weibo_dataset/train_file.txt"
we_test_csv_name = "weibo_dataset/test_file.txt"

## Image graph data (Node Embeddings)
we_image_vec_dir = "weibo_data/image_data/"

## Text graph data (Node Embeddings)
we_text_vec_dir = "weibo_data/text_data/"


######## Parameters ######## 
batch_size = 64
epochs = 10
lr = 1e-5
gradient_accumulation_steps = 2