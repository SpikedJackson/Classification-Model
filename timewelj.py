import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# a dictionary that maps HARPNUM to NOAA_ARS
HARP = {}

# process txt file
with open("datasets/data-2010-15/all_harps_with_noaa_ars.txt") as file:
    # skip header
    next(file)
    # parse line by line
    for line in file:
        # remove all new line and whitespace
        line = line.strip('\n')
        line = line.split(' ')
        line = list(filter(None, line))
        line_ = []
        for thing in line:
            line_.append(thing.split(','))
        # map HARPNUM value to array of NOAA_ARS values
        HARP[line_[0][0]] = line_[1]

# an array with 8 columns and 660 rows, each row contains details about a GOES event
goes = np.load("datasets/data-2010-15/goes_data.npy", allow_pickle=True)

# an array with 3 columns and 315 rows, columns are: HARPNUM, Peak Flare Time, Class of Energy Burst    
posClass = np.load("datasets/data-2010-15/pos_class.npy")

# an array with 3 columns and 315 rows, columns are: HARPNUM, Peak Flare Time, Class of Energy Burst    
negClass = np.load("datasets/data-2010-15/neg_class.npy", allow_pickle=True)

# an array with 3 columns and 630 rows, combining the previous two arrays
allClass = np.concatenate((posClass, negClass), axis=0)

class my_svm():

    def __init__(self,year):
        # initalize variables
        self.C = 0.001
        self.epochs = 500
        self.learning_rate = 0.01

        # an array with 90 columns and 315 rows, columns 1 - 18 are the features of positive solar events
        pos = np.load("datasets/data-" + year + "/pos_features_main_timechange.npy")

        # an array with 90 columns and 315 rows, columns 1 - 18 are the features of negative solar events
        neg = np.load("datasets/data-" + year + "/neg_features_main_timechange.npy")

        # an array with 1 column and 315 rows, column 1 is an additional feature of positive solar events
        posHist = np.load("datasets/data-" + year + "/pos_features_historical.npy")

        # an array with 1 column and 315 rows, column 1 is an additional feature of negative solar events
        negHist = np.load("datasets/data-" + year + "/neg_features_historical.npy")

        # an array with 18 columns and 315 rows, columns are additional features of positive solar events
        posMaxMin = np.load("datasets/data-" + year + "/pos_features_maxmin.npy", allow_pickle=True)

        # an array with 18 columns and 315 rows, columns are additional features of negative solar events
        negMaxMin = np.load("datasets/data-" + year + "/neg_features_maxmin.npy", allow_pickle=True)

        # an array of indices, this is the order we should train the model in (so start with 485th observation, then 575th)
        self.order = np.load("datasets/data-" + year + "/data_order.npy")

        # organize data for preprocess
        self.data = [[pos[:,:18], neg[:,:18]], [pos[:,18:], neg[:,18:]], [posHist, negHist], [posMaxMin, negMaxMin]]

    def preprocess(self,):
        # combine all data for normalization
        processed_data = (np.concatenate((np.concatenate((self.data[0][0],self.data[0][1]),axis=0),np.concatenate((self.data[1][0],self.data[1][1]),axis=0),np.concatenate((self.data[2][0],self.data[2][1]),axis=0),np.concatenate((self.data[3][0],self.data[3][1]),axis=0)),axis=1))
        
        # using StandardScaler to normalize the input
        scalar = StandardScaler().fit(processed_data)
        data_normalized = scalar.transform(processed_data)

        # remove missing values
        data_normalized = data_normalized[~np.isnan(data_normalized).any(axis=1)]
        
        # assign labels to target (1 and -1)
        target = [np.ones((len(self.data[0][0]),1)), -np.ones((len(self.data[0][1]),1))]
        
        # un-combine all data back to feature sets, return with target
        return [data_normalized[:,:18], data_normalized[:,18:90], data_normalized[:,90:91], data_normalized[:,91:]], target
    
    def feature_creation(self, fs_value):
        # preprocess data
        processed, target = self.preprocess()

        # handle given feature set label
        if fs_value == "FS-I":
            i = 0
        elif fs_value == "FS-II":
            i = 1
        elif fs_value == "FS-III":
            i = 2
        elif fs_value == "FS-IV":
            i = 3

        # create 2 D array of corresponding features
        bias = np.ones(len(processed[i]))
        return np.column_stack((processed[i], bias)), np.concatenate((target[0], target[1]), axis = 0)
    
    def feature_combination_creator(self, fs_values):
        # start by creating first feature set
        X,Y = self.feature_creation(fs_values[0])

        # for every addition feature set, concatenate together
        for set in fs_values[1:]:
            X = np.concatenate((X,self.feature_creation(set)[0]),axis=1)
        
        # return combined feature set
        return X,Y
    
    def cross_validation(self,X,Y, feature_set):
        # reorder data
        X, Y = self.shuffle_order(X,Y)
        self.weights = np.zeros(X.shape[1])

        accuracies = []
        best_y_true = []
        best_y_pred = []
        best_acc = -1

        # use k-fold with k=10
        k_fold = KFold(n_splits=10)
        for train, test in k_fold.split(X):
            # split the data into train and test sets
            x_train, x_test = X[train], X[test]
            y_train, y_test = Y[train], Y[test]

            # call training function
            self.training(x_train, y_train)

            # call predict function
            y_pred = self.predict(x_test)

            # call tss function
            accuracy = self.tss(y_test, y_pred)

            # save values for averaging and best fold for confusion matrix
            accuracies.append(accuracy)
            if accuracy > best_acc:
                best_y_true = y_test.ravel()
                best_y_pred = y_pred
                best_acc = accuracy

        # create chart for visualizing
        # set the plot and plot size
        fig, ax = plt.subplots()
        fig.set_size_inches((15,8))

        # plot points
        ax.plot(range(10),accuracies)

        # set the x-labels
        ax.set_xlabel("Fold")

        # set the y-labels
        ax.set_ylabel("TSS Score")
        ax.set_ylim([-1,1])
        
        # set the title
        ax.set_title("TSS Score per Fold using Feature Set " + str(feature_set))

        # show the plot
        plt.show()

        # display confusion matrix for best fold
        disp = ConfusionMatrixDisplay(confusion_matrix(best_y_true,best_y_pred, labels=[-1,1]),display_labels=[-1,1])
        disp.plot()
        plt.title("Confusion Matrix for Feature Set " + str(feature_set))
        plt.show()

        # output average accuracy across all train test splits
        return sum(accuracies)/len(accuracies)
    
    def shuffle_order(self,X,Y):
        new_x = np.full_like(X,0)
        new_y = np.full_like(Y,0)
        for j,num in enumerate(self.order):
            new_x[j] = X[num]
            new_y[j] = Y[num]
        return new_x, new_y
    
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y* np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equalt to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance
    
    def training(self, X, Y):
        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epochs):
            for i, feature in enumerate(X):
                gradient = self.compute_gradient(feature, Y[i])
                self.weights = self.weights - (self.learning_rate * gradient)
    
    def predict(self,X):
        # compute predictions on test set
        return [np.sign(np.dot(X[i], self.weights)) for i in range(X.shape[0])]

    def tss(self,true,predict):
        # count true/false positive/negative
        TP,FN,FP,TN = 0,0,0,0
        for i, y in enumerate(predict):
            if y == true[i]:
                if y == 1:
                    TP += 1
                if y == -1:
                    TN += 1
            elif y == 1:
                FP += 1
            else:
                FN += 1

        # compute true skill score
        return (TP/(TP+FN))-(FP/(FP+TN))
    
def power_set(s):
    # compute all combinations given a list of feature sets
    x = len(s)
    total = []
    for i in range(1, 1 << x):
        total.append([s[j] for j in range(x) if (i & (1 << j))])
    return total
    
def feature_experiment():
    # train (and test) on 2010 dataset
    svm = my_svm("2010-15")
    feature_sets = power_set(["FS-I","FS-II","FS-III","FS-IV"])
    max_tss = [-1,"NONE"]

    # test on all 4 feature set combinations
    for combination in feature_sets:

        # combine feature sets
        X,Y = svm.feature_combination_creator(combination)
        
        # perform cross validation (also includes confusion matrix and plot showing tss scores per fold)
        performance = svm.cross_validation(X,Y,combination)

        # TSS average scores for k-fold validation printed out on console
        print("Average TSS score on " + str(combination) + ":", performance)

        # save the best performing feature set combination
        if performance > max_tss[0]:
            max_tss = [performance, combination]
    
    # print the best performing feature set combination
    print("Best performing feature set:", str(max_tss[1]))
    return max_tss[1]

def data_experiment(feature_combo = ['FS-IV']):
    # use the best performing feature combination from feature_experiment on 2010 data
    svm_2010 = my_svm("2010-15")

    # combine feature sets
    X,Y = svm_2010.feature_combination_creator(feature_combo)

    # perform cross validation (also includes confusion matrix and plot showing tss scores per fold)
    performance_2010 = svm_2010.cross_validation(X,Y, feature_combo)

    # TSS average scores for k-fold validation printed out on console
    print("Average TSS score on 2010:", performance_2010)

    # use the best performing feature combination from feature_experiment on 2010 data
    svm_2020 = my_svm("2020-24")

    # combine feature sets
    X,Y = svm_2020.feature_combination_creator(feature_combo)

    # perform cross validation (also includes confusion matrix and plot showing tss scores per fold)
    performance_2020 = svm_2020.cross_validation(X,Y, feature_combo)

    # TSS average scores for k-fold validation printed out on console
    print("Average TSS score on 2020:", performance_2020)

best_set = feature_experiment()
# best_set is "FS-IV"
data_experiment()
