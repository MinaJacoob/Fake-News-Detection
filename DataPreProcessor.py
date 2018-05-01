import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreProcessor:
    DATA_PATH = ""
    stop_words = set(stopwords.words("english"))

    def __init__(self, PATH):
        self.DATA_PATH = PATH

    def LoadData(self):
        self.df = pd.read_csv(self.DATA_PATH)
        return self.df

    def PrepareData(self):
        try:
            self.df = self.df.drop(['id'], axis=1) # drop id
            self.df = self.df.drop(['author'], axis=1) #drop author
            self.df['content'] = self.df['title'].astype(str) + df['text'] # merge title with text into new column called content
            self.df = self.df.drop(['title'],axis=1) #drop title column
            self.df = self.df.drop(['text'],axis=1) #drop text column
            self.df.to_csv("out.csv",index=False) #save new dataframe as out.csv
            print(self.df.head())
        except:
            print("the data is already preprocessed")

    def PreProcess(self):
        label = []
        content = []
        for sentence in range(0, len(self.df)):
            content_data = str(self.df.iloc[sentence][1]).lower()
            label_data = self.df.iloc[sentence][0]
            tokenized_words = word_tokenize(content_data)
            filtered_words = [w for w in tokenized_words if not w in self.stop_words]
            label.append(label_data)
            content.append(filtered_words)
            tokenized_words = []
            filtered_words = []
        modified_dataFrame = pd.DataFrame({'label':label, 'content':content})
        modified_dataFrame.to_csv("PreProcessedData.csv",index=False)

