import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sklearn.metrics as skm

if __name__ == "__main__":
    # Load the npy files for train, val and test sets
    x_train = np.load('train_val_test_sets/x_train.npy')
    y_train = np.load('train_val_test_sets/y_train.npy')

    x_val = np.load('train_val_test_sets/x_val.npy')
    y_val = np.load('train_val_test_sets/y_val.npy')

    x_test = np.load('train_val_test_sets/x_test.npy')
    y_test = np.load('train_val_test_sets/y_test.npy')

    # One vs All function. Sorry I named it this way because I am a huge fan of boku no hero
    def one_for_all(y_arr, the_one):
        # y_arr: your labels array input
        # the_one: name of the class that will be the one

        ## Class 0 will be the All, Specified class will be the One
        new_y = (y_arr == the_one) + 0

        return new_y

    ## Function for getting class wise accuracy
    def class_wise_accuracy(actual_y, pred_y, label):
        # actual_y: actual y values
        # pred_y: predicted y values
        # label: class

        acc = (1/np.sum(actual_y == label)) * (np.sum((actual_y == label)*(pred_y == label)))

        return acc

    ## Function for vanilla accuracy
    def vanilla_taste_good(actual_y, pred_y, num_inst):
        # actual_y: actual y values
        # pred_y: predicted y values
        # num_inst: number of instances

        acc = np.sum(pred_y == actual_y) / num_inst

        return acc

    ## Function for converting labels in an array to their respective digit labels using a dictionary for mapping
    def season_map(y, mapper):
        # y: your labels array
        # mapper: your dictionary used for mapping

        new_y = np.zeros((y.shape))
        for ind, label in enumerate(y):
            new_y[ind] = mapper[label]

        return new_y

    # ## Finding the best C

    ## Collate the class-wise averaged accuracy for all regulatisation constants over all seasons
    reg_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]
    seasons = ['spring', 'summer', 'autumn', 'winter']

    # A matrix to contain the scores of each constans' score for each season, 4 x 7 dimension
    reg_matrix = np.zeros((len(seasons), len(reg_constants)))

    ## For each constant, test which each season and append the scores to the matrix
    for ind, con in  enumerate(reg_constants):
        acc_scores = [] # To store the accuracy values for each season
        for season in seasons:
            # We change the labels for each season to do a One vs All SVM
            season_v_all_train = one_for_all(y_train, season)
            season_v_all_val = one_for_all(y_val, season)

            # Set up the model
            clf = SVC(C=con, kernel='linear', probability=True)
            clf.fit(x_train, season_v_all_train)

            # Predict with validation set
            pred_y = clf.predict_proba(x_val)
            pred_y = np.argmax(pred_y, axis=1)

            # Use class wise accuracy function to calculate accuracy score. We use 1 here as it represents the current
            # season's label
            acc_score = class_wise_accuracy(season_v_all_val, pred_y, 1)
            acc_scores.append(acc_score)

        reg_matrix[:, ind] = acc_scores

    ## Next we average the values in the matrix to check which constant performed the best over all seasons
    avg = np.average(reg_matrix, axis=0)
    for ind, x in enumerate(reg_constants):
        print(f'The class-wise average accuracy over the 4 seasons for c={reg_constants[ind]} is {avg[ind]}')

    # In this case we get 0.01 as the best regularisation constant
    best_c = reg_constants[np.argmax(avg)]
    print(f'Overall, the best class-wise average accuracy was with c={best_c}')

    # # Now to use our Constant to test against the combination of train and val

    # We assign the class labels 1,2,3,4 to spring,summer,autumn,winter respectively
    s_dict = {'spring':1, 'summer':2, 'autumn':3, 'winter':4}

    # Combine the training and validation sets
    x_train_val = np.concatenate((x_train, x_val))
    y_train_val = np.concatenate((y_train, y_val))

    # ### Use class-wise average metrics to get accuracy

    # Matrix for storing the True probabilities for each season.
    # This will be used to find our final predicted labels for vanilla accuracy test. Dimension of len(y_test) x len(seasons)
    fin_y_pred = np.zeros((len(y_test), len(seasons)))

    # To store class-wise accuracy scores
    class_wise_scores = []

    # One Vs All Probabilities
    for ind, season in enumerate(seasons):
        # We change the labels for each season to do a One vs All SVM
        season_v_all_train_val = one_for_all(y_train_val, season) # train validation labels
        season_v_all_test = one_for_all(y_test, season) # test labels

        # Model
        clf = SVC(C=best_c, kernel='linear', probability=True)
        clf.fit(x_train_val, season_v_all_train_val)

        # Predict probability and take the values that correspond to True
        pred_prob = clf.predict_proba(x_test)
        sea_y_pred = pred_prob[:,1]

        # Append to fin_y_pred
        fin_y_pred[:,ind] = sea_y_pred

        # Use argmax to get the predictions
        pred_y = np.argmax(pred_prob, axis=1)

        # Get class wise accuracy and append
        acc = class_wise_accuracy(season_v_all_test, pred_y, 1)
        class_wise_scores.append(acc)

    class_wise_average_accuracy = np.average(class_wise_scores)
    print(f'The class-wise averaged accuracy for c={best_c} is {class_wise_average_accuracy}')

    # Next we use argmax in the y axis, which gives us a matrix of len(y_test) x 1
    # And this represents the most probable labels class wise
    fin_y_pred = np.argmax(fin_y_pred, axis=1) + 1

    # # Now to test our labels with class-wise  and vanilla accuracy
    map_to_num = season_map(y_test, s_dict)

    vanilla_acc = vanilla_taste_good(map_to_num, fin_y_pred, len(y_test))
    print(f'The vanilla accuracy for c={best_c} is {vanilla_acc}')


