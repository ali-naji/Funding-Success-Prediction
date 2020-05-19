import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class Pipeline:
    def __init__(self, NUM_VARS, CAT_VARS, STR_VARS, BOOL_VARS, DROP_VARS, TARGET, random_state, alpha=0.0002, train_size=0.7,
                 test_size=0.33, rare_perc=0.01, percentile=95.0, max_features=1000):

        # datasets
        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        # data types
        self.NUM_VARS = NUM_VARS
        self.CAT_VARS = CAT_VARS
        self.STR_VARS = STR_VARS
        self.BOOL_VARS = BOOL_VARS
        self.DROP_VARS = DROP_VARS
        self.TARGET = TARGET

        # General parameters
        self.random_state = random_state
        self.train_size = train_size
        self.test_size = test_size
        self.rare_perc = rare_perc
        self.percentile = percentile

        # data properties
        self.frequent_labels_dict = {}
        self.num_percentiles = {}
        self.TEXT_COL = 'kw_desc'
        self.selected_features = None

        # Models
        self.encoder = OneHotEncoder(sparse=False)
        self.selector = SelectFromModel(
            Lasso(random_state=random_state, alpha=alpha))
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
                                          ngram_range=(1, 2), max_df=0.7, max_features=max_features)
        self.scaler = MinMaxScaler()
        self.model = LinearSVC(random_state=random_state,
                               C=1, loss='hinge', penalty='l2')

    ############################# functions that extract information from train set ############################
    def get_frequent_labels(self):
        for var in self.CAT_VARS:
            tmp = self.X_train.groupby(
                var)[self.TARGET].count()/len(self.X_train)
            self.frequent_labels_dict[var] = tmp[tmp > self.rare_perc].index
        return self

    def get_ohe(self):
        self.encoder.fit(self.X_train[self.CAT_VARS])
        return self

    def get_num_percentiles(self):
        for var in self.NUM_VARS:
            (p1, p2) = np.percentile(
                self.X_train[var], ((100-self.percentile)/2, (100+self.percentile)/2))
            self.num_percentiles[var] = (p1, p2)
        return self

    def train_vectorizer(self):
        self.vectorizer.fit(self.X_train[self.TEXT_COL])
        return self

    def train_scaler(self):
        self.scaler.fit(self.X_train[self.X_train.columns])
        return self

    def train_selector(self):
        self.selector.fit(self.X_train, self.y_train)
        self.selected_features = list(
            self.X_train.columns[self.selector.get_support()])
        return self

    ############################# functions that transform new sets ############################
    def drop_na(self, df):
        df = df.copy()
        df = df.dropna().reset_index(drop=True)
        return df

    def drop_vars(self, df):
        df = df.copy()
        df = df.drop(self.DROP_VARS, axis=1, errors='ignore')
        return df

    def split_data(self, df):
        df = df.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df, df[self.TARGET], random_state=self.random_state, train_size=self.train_size)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test, self.y_test, random_state=self.random_state, test_size=self.test_size)
        return self

    def apply_rare_labels(self, df):
        df = df.copy()
        for var in self.CAT_VARS:
            df[var] = np.where(df[var].isin(
                self.frequent_labels_dict[var]), df[var], 'Rare')
        return df

    def apply_ohe(self, df):
        df = df.copy()
        encoded_catgs_df = pd.DataFrame(self.encoder.transform(
            df[self.CAT_VARS]), columns=np.concatenate(self.encoder.categories_).ravel())
        df = pd.concat([df.reset_index(drop=True), encoded_catgs_df], axis=1).drop(
            self.CAT_VARS+['Rare'], axis=1)
        return df

    def apply_bool2int(self, df):
        df = df.copy()
        for var in self.BOOL_VARS:
            df[var] = df[var].apply(lambda x: 1 if bool(x) else 0)
        return df

    def apply_num_percentiles(self, df):
        df = df.copy()
        for var in self.NUM_VARS:
            df = df.loc[df[var].between(
                self.num_percentiles[var][0], self.num_percentiles[var][1]), :]
            df = df.reset_index(drop=True)
        return df

    def merge_strings(self, df, vars=['keywords', 'desc']):
        df = df.copy()
        df[self.TEXT_COL] = (df[vars[0]] + ' ' +
                             df[vars[1]]).str.replace('-', ' ')
        df.drop(vars, axis=1, inplace=True)
        return df

    def apply_text_transform(self, df):
        df = df.copy()
        feat_names = self.vectorizer.get_feature_names()
        transformed_mat = pd.DataFrame(self.vectorizer.transform(
            df[self.TEXT_COL]).todense().tolist(), columns=feat_names)
        df = pd.concat([df, transformed_mat], axis=1).drop(
            self.TEXT_COL, axis=1)
        return df

    def apply_scaler(self, df):
        df = df.copy()
        df[df.columns] = self.scaler.transform(df[df.columns])
        return df

    def subset_features(self, df):
        df = df.copy()
        df = df[self.selected_features]
        return df

    ############################# function that runs fitting on training data ############################

    def fit(self, data):
        print("Started Training Pipeline")
        data = self.drop_na(data)
        data = self.drop_vars(data)
        self.split_data(data)

        print("Applying rare labels")
        self.get_frequent_labels()
        self.X_train = self.apply_rare_labels(self.X_train)
        self.X_val = self.apply_rare_labels(self.X_val)
        self.X_test = self.apply_rare_labels(self.X_test)

        print("Applying One-Hot Encoding")
        self.get_ohe()
        self.X_train = self.apply_ohe(self.X_train)
        self.X_val = self.apply_ohe(self.X_val)
        self.X_test = self.apply_ohe(self.X_test)

        print("Converting Bools to Ints")
        self.X_train = self.apply_bool2int(self.X_train)
        self.X_val = self.apply_bool2int(self.X_val)
        self.X_test = self.apply_bool2int(self.X_test)

        print("Filtering Outliers")
        self.get_num_percentiles()
        self.X_train = self.apply_num_percentiles(self.X_train)
        self.X_val = self.apply_num_percentiles(self.X_val)
        self.X_test = self.apply_num_percentiles(self.X_test)

        print("Engineering Text Column")
        self.X_train = self.merge_strings(self.X_train)
        self.X_val = self.merge_strings(self.X_val)
        self.X_test = self.merge_strings(self.X_test)

        self.train_vectorizer()

        self.X_train = self.apply_text_transform(self.X_train)
        self.X_val = self.apply_text_transform(self.X_val)
        self.X_test = self.apply_text_transform(self.X_test)

        print("Popping Target Variable")
        self.y_train = self.X_train.pop(self.TARGET)
        self.y_val = self.X_val.pop(self.TARGET)
        self.y_test = self.X_test.pop(self.TARGET)

        print("Applying Scaler")
        self.train_scaler()
        self.X_train = self.apply_scaler(self.X_train)
        self.X_val = self.apply_scaler(self.X_val)
        self.X_test = self.apply_scaler(self.X_test)

        print("Subsetting Features")
        self.train_selector()
        self.X_train = self.subset_features(self.X_train)
        self.X_val = self.subset_features(self.X_val)
        self.X_test = self.subset_features(self.X_test)

        print("Training ML Model")
        self.model.fit(self.X_train, self.y_train)

        print("Finished Training Pipeline")

        return self

    def transform(self, data):
        data = self.drop_na(data)
        data = self.drop_vars(data)
        data = self.apply_rare_labels(data)
        data = self.apply_ohe(data)
        data = self.apply_bool2int(data)
        data = self.apply_num_percentiles(data)
        data = self.merge_strings(data)
        data = self.apply_text_transform(data)
        if self.TARGET in data.columns:
            data.drop(self.TARGET, axis=1, inplace=True)
        data = self.apply_scaler(data)
        data = self.subset_features(data)
        return data

    def predict(self, data):
        data = self.transform(data)
        return self.model.predict(data)

    def evaluate(self):
        print("Training Set Accuracy = "+str(accuracy_score(self.y_train,
                                                            self.model.predict(self.X_train)))+"%")
        print("Training Set ROC-AUC = "+str(roc_auc_score(self.y_train,
                                                          self.model.predict(self.X_train)))+"%")
        print()
        print("Validation Set Accuracy = "+str(accuracy_score(self.y_val,
                                                              self.model.predict(self.X_val)))+"%")
        print("Validation Set ROC-AUC = "+str(roc_auc_score(self.y_val,
                                                            self.model.predict(self.X_val)))+"%")
