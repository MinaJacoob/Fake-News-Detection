from sklearn.model_selection import train_test_split

class ModelPreparation:
    def __init__(self, dataFrame):
        self.df = dataFrame

    def SplitData(self):
        X, Y = self.df['content'], self.df['label']
        train, test = train_test_split(self.df, test_size = 0.2)
        print(train.shape)
        print(test.shape)