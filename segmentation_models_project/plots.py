import matplotlib.pyplot as plt



# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def log_training(df:dict, train_logs: dict, validation_logs:dict):
    train_logs = {"train_"+k:v for k,v in train_logs.items()}
    validation_logs = {"val_"+k:v for k,v in validation_logs.items()}
    
    for k,v in train_logs.items():
        df[k].append(v)
    for k,v in validation_logs.items():
        df[k].append(v)
    return df

def create_log_dict(metrics, loss):
    df_train = {"train_"+metric_name.__name__:[] for metric_name in metrics}
    df_train["train_loss"] = [] # Loss is only named loss. Its type can be found in the filename
    # df_train["train_"+loss.__name__] = []
    df_val = {"val_"+metric_name.__name__:[] for metric_name in metrics}
    df_val["val_loss"] = []
    # df_val["val_"+loss.__name__] = []
    df_train.update(df_val)
    return df_train