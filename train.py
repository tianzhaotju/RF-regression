import warnings
from RF import RandomForest
if __name__ == "__main__":
    # 忽略一些版本不兼容等警告
    warnings.filterwarnings("ignore")
    orf = RandomForest(input=722)
    orf.offline_train()
    orf.online_train()
# RMSE: 0.036099242494643385
# Score:-2.9361148161707598
