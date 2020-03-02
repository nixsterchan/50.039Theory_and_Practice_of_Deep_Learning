import numpy as np
import pandas as pd
import sklearn

if __name__ == "__main__":
    # ### Function for splitting
    def train_val_test(arr, train_per, val_per):
        # arr: your array
        # train_per: train percentage
        # val_per: validation percentage
        arr_len = len(arr)
        # first index to cut at
        first = int(np.floor(train_per / 100 * arr_len))
        # second index to cut at
        second = int(np.floor((train_per + val_per) / 100 * arr_len))

        ## Split up the arr
        train_s = arr[0:first]
        val_s = arr[first:second]
        test_s = arr[second:arr_len]

        return train_s, val_s, test_s

    # ### Load the concepts file to get respective labels
    concepts = pd.read_csv('concepts_2011.txt', sep='\t')

    ## Get indexes of
    names = concepts['name'].to_numpy()
    # short for seasonal indexes
    s_inds = {}

    for ind, n in enumerate(names):
        if n == 'Summer':
            s_inds[n] = ind
        if n == 'Autumn':
            s_inds[n] = ind
        if n == 'Spring':
            s_inds[n] = ind
        if n == 'Winter':
            s_inds[n] = ind

    # ### Create table with headers ranging from index 0 to 98
    # Get the column names
    indexes = range(len(concepts))
    img_name = 'image'
    column_names = [img_name] + list(indexes)

    # Load the trainset annotations file
    trainset = pd.read_csv('trainset_gt_annotations.txt', header=None, names=column_names, sep=' ')

    # Filter out the images that dont fall within seasons
    season_imgs = trainset[(trainset[s_inds['Spring']] == 1) | (trainset[s_inds['Summer']] == 1) | (trainset[s_inds['Autumn']] == 1) | (trainset[s_inds['Winter']] == 1)]
    season_imgs = season_imgs.reset_index().drop(columns='index')

    ## Now select the columns that matter
    season_imgs = season_imgs[[img_name, s_inds['Spring'], s_inds['Summer'], s_inds['Autumn'], s_inds['Winter']]]

    s_arr = season_imgs.to_numpy()

    spring, summer, autumn, winter = [], [], [], []

    for x in s_arr:
        if x[1] == 1:
            spring.append([x[0], 'spring'])
        elif x[2] == 1:
            summer.append([x[0], 'summer'])
        elif x[3] == 1:
            autumn.append([x[0], 'autumn'])
        elif x[4] == 1:
            winter.append([x[0], 'winter'])


    # # Now to build our training, validation and test sets class-wise with 60, 15 and 25 percent respectively

    # Use splitting function to split accordingly
    spring_split = train_val_test(spring, 60, 15)
    summer_split = train_val_test(summer, 60, 15)
    autumn_split = train_val_test(autumn, 60, 15)
    winter_split = train_val_test(winter, 60, 15)

    # Concat the values for train val and test into temporary lists
    train_temp = spring_split[0] + summer_split[0] + autumn_split[0] + winter_split[0]
    val_temp = spring_split[1] + summer_split[1] + autumn_split[1] + winter_split[1]
    test_temp = spring_split[2] + summer_split[2] + autumn_split[2] + winter_split[2]

    ### Load the image npy files into each respective set, and save as npy file
    # Train Set
    x_train = np.empty((0,1024))
    y_train = []
    for x in train_temp:
        x_train = np.append(x_train, [np.load(f'imageclef2011_feats/{x[0]}_ft.npy')], axis=0)
        y_train.append(x[1])
    y_train = np.array(y_train)

    np.save(f'train_val_test_sets/x_train.npy', x_train)
    np.save(f'train_val_test_sets/y_train.npy', y_train)

    # Validation Set
    x_val = np.empty((0,1024))
    y_val = []
    for x in val_temp:
        x_val = np.append(x_val, [np.load(f'imageclef2011_feats/{x[0]}_ft.npy')], axis=0)
        y_val.append(x[1])
    y_val = np.array(y_val)

    np.save(f'train_val_test_sets/x_val.npy', x_val)
    np.save(f'train_val_test_sets/y_val.npy', y_val)

    # Test Set
    x_test = np.empty((0,1024))
    y_test = []
    for x in test_temp:
        x_test = np.append(x_test, [np.load(f'imageclef2011_feats/{x[0]}_ft.npy')], axis=0)
        y_test.append(x[1])
    y_test = np.array(y_test)

    np.save(f'train_val_test_sets/x_test.npy', x_test)
    np.save(f'train_val_test_sets/y_test.npy', y_test)

