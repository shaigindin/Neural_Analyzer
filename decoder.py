# -=-=-=-=-=-=--=-=-=-= IMPORTS =-=-=-=-=-=-=--
import pandas as pd
from scipy.ndimage import gaussian_filter
import fnmatch
import re
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import random
import pickle
from scipy.io import loadmat
from typing import List
from pandas import *
import numpy as np
import math
import mat4py as mp
from tqdm import tqdm


ALL_POSSIBILE_POPULATIONS = ["SNR", "MSN", "TAN", "CS","SS","CRB"]
REG_FOR_FRAGMENTS = r'^(CRB|MSN|SS|CS|TAN|SNR)\d+'
MSG = "Choose folders to cnvert its files to csv"

RUNNING_THE_TEST = "Choose folders for running the test"

LAST_COLUMN = -1

class decoder(object):
    """
    Decoder Class
    """
    NUMBER_OF_ITERATIONS = 100  # number of iteration of each group of cells for finding a solid average
    SIGMA = 30  # sigma for the gaussian
    NEIGHBORS = 1  # only closet neighbour, act like SVM
    TIMES = 30  # number of iteration on each K-population of cells.
    K = 48  # number of files per time
    LAG = 1000  # where to start the experiment (in the eye movement)
    d = {0: "PURSUIT", 1: "SACCADE"} # innder dictionary
    SEGMENTS = 12 #how many segment of 100ms we want to cut.
    SAMPLES_LOWER_BOUND = 100  # filter the cells with less than _ sampels
    number_of_cells_to_choose_for_test = 1 #when buildin X_test matrice, how many samples from each direction / reward
    step = 1
    __algo_names = ["simple_knn", "simple_knn_fregments"]


    def __init__(self, input_dir: str, output_dir: str, population_names: List[str]):
        """
        insert valid input_dir, output_dir and the population name mus be on
        @param input_dir:
        @param output_dir:
        @param population_names: must be from msn CRB ss cs SNR (mabye more..)
        """
        self.__input_dir = os.path.join(input_dir, '')
        self.__output_dir = os.path.join(output_dir, '')
        self.population_names = [x.upper() for x in population_names]
        self.__temp_path_for_writing = output_dir
        self.__files = dict()
        self.ALGOS = {
            "simple_knn": self.simple_knn,
            "simple_knn_fregments": self.simple_knn_fragments
        }

    @staticmethod
    def get_population_name(cell_name):
        """
        gets 39#CRB_4847.csv
        return CRB
        """
        return cell_name[cell_name.find("#")+1:cell_name.find("_")]


    @staticmethod
    def get_population_name_and_population(cell_name):
        """
        gets 39#CRB_4847.csv
        return CRB_4847
        """
        return cell_name[cell_name.find("#")+1:cell_name.find(".")]

    @staticmethod
    def get_cell_name(cell_name):
        """
        gets 39#CRB_4847.csv
        return 4847
        """
        return cell_name[cell_name.find("_") + 1:cell_name.find(".")]

    @staticmethod
    def get_acc_df_for_graph(file_paths:List, time=-1):
        """
        get list of folders or files and makes a whole data frame
        """
        time_list = []
        algo_name_list = []
        kind_name_list = []
        rate_list = []
        population_name_list = []
        K_population = []
        expirement_list = []
        group=[]
        stddev = []
        for file_path in file_paths:
                if os.path.isdir(file_path):
                    cell_names = fnmatch.filter(os.listdir(file_path), '*')
                    cell_names = [name for name in cell_names if name in ALL_POSSIBILE_POPULATIONS]
                    # cell_names = fn.filter(cell_names, ALL_POSSIBILE_POPULATIONS)
                    file_path = os.path.join(file_path, '')
                    cell_names = [file_path + name for name in cell_names]
                elif os.path.isfile(file_path):
                    cell_names = [file_path]
                else:
                    print("file path is not valid")
                    exit(1)
                for file_name_path in cell_names:
                    with open(file_name_path, 'rb') as info_file:
                        info = pickle.load(info_file)
                        for i,tup in enumerate(info):
                            if i==0 and time == -1:
                                K_population.append(i+1)
                                acc_list = [result[1] for result in tup[0]]
                            else:
                                K_population.append(len(tup[0][0][0][0]))
                                acc_list = [result[0][1] for result in tup[0]]
                            deviation = np.std(np.array(acc_list), ddof=1) / math.sqrt(len(acc_list))
                            stddev.append(deviation)
                            rate_list.append(tup[1])
                            time_list.append(time)
                            name = os.path.basename(file_name_path)
                            population_name_list.append(name)
                            algo_name = os.path.basename(os.path.dirname(file_name_path))
                            algo_name_list.append(algo_name)
                            kind_name = os.path.basename(os.path.dirname(os.path.dirname(file_name_path)))
                            kind_name_list.append(kind_name)
                            expirement_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_name_path))))
                            expirement_list.append(os.path.basename(expirement_name))
                            group.append("\n".join([expirement_name,kind_name,algo_name, name]))
        return DataFrame({'concatenated_cells': K_population, 'acc': rate_list,
                              'population': population_name_list,'kind':kind_name_list, 'algorithm':algo_name_list,
                              'experiment': expirement_list, 'group': group, 'std': stddev, 'time': time_list})

    @staticmethod
    def get_acc_df_for_graph_frag(file_paths: List):
        time_list = []
        algo_name_list = []
        kind_name_list = []
        rate_list = []
        population_name_list = []
        K_population = []
        expirement_list = []
        group = []
        stddev = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                cell_names = fnmatch.filter(os.listdir(file_path), '*')
                cell_names = [os.path.join(file_path, val) for val in cell_names if re.search(REG_FOR_FRAGMENTS, val)]
            elif os.path.isfile(file_path):
                cell_names = [file_path]
            else:
                print("file path is not valid")
                exit(1)
            for file_name_path in cell_names:
                with open(file_name_path, 'rb') as info_file:
                    info = pickle.load(info_file)
                    for i, tup in enumerate(info):
                        acc_list = [val[0][1] for val in tup[0]]
                        # print(len(tup[0][0][0][0]))
                        K_population.append(len(tup[0][0][0][0]))
                        acc_list = [result[0][1] for result in tup[0]]
                        deviation = np.std(np.array(acc_list), ddof=1) / math.sqrt(len(acc_list))
                        stddev.append(deviation)
                        rate_list.append(tup[1])
                        name = os.path.basename(file_name_path)
                        time = int(''.join(i for i in name if i.isdigit()))
                        time_list.append(time)
                        name = ''.join(i for i in name if i.isalpha())
                        population_name_list.append(name)
                        algo_name = os.path.basename(os.path.dirname(file_name_path))
                        algo_name_list.append(algo_name)
                        kind_name = os.path.basename(os.path.dirname(os.path.dirname(file_name_path)))
                        kind_name_list.append(kind_name)
                        expirement_name = os.path.basename(
                            os.path.dirname(os.path.dirname(os.path.dirname(file_name_path))))
                        expirement_list.append(os.path.basename(expirement_name))
                        group.append("\n".join([expirement_name, kind_name, algo_name, name]))
        return DataFrame({'concatenated_cells': K_population, 'acc': rate_list,
                          'population': population_name_list, 'kind': kind_name_list, 'algorithm': algo_name_list,
                          'experiment': expirement_list, 'group': group, 'std': stddev, 'time': time_list})




    @staticmethod
    def get_population_one_cell_data_frame(file_path:str):
        """
        get file path for example
        ~/MATY/Neural_Analyzer/out/nogas_project/target_direction/pursuit/simple_knn
        and return data frame of the cell names and their accuracy
        """
        if os.path.isdir(file_path):
            cell_names = fnmatch.filter(os.listdir(file_path), '*')
            cell_names = [name for name in cell_names if name in ALL_POSSIBILE_POPULATIONS]
            file_path= os.path.join(file_path, '')
            cell_names = [file_path + name for name in cell_names]
        elif os.path.isfile(file_path):
            cell_names = [file_path]
        names_list = []
        rate_list = []
        population_list = []
        for file_name_path in cell_names:
            with open(file_name_path, 'rb') as info_file:
                info = pickle.load(info_file)
                for tup in info[0][0]:
                    names_list.append(decoder.get_cell_name(tup[0]))
                    rate_list.append(tup[1])
                    population_list.append(decoder.get_population_name(tup[0]))
        return DataFrame({'cell_name':names_list, 'acc':rate_list, 'type':population_list})


    def filter_cells(self, cell_names, name):
        """
        remove from list the names which not conatin name string
        @param cell_names: list of the cell names
        @param name: SNR/msn/cs/.. etc
        @return:
        """
        return list(
            filter(lambda cell_name: True if cell_name.find(name) != -1 else False, [x.split(".")[0].upper() + "." + x.split(".")[1] for x in cell_names]))


    def ask_for_dirs(self, path: str, msg):
        """
        load question for user and return list of ints represent the user's choices
        """
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        print(msg)
        for i,folder in enumerate(subfolders):
            print(i+1,") ", folder)
        input_string = input("enter all the folders number with space between them\n")
        userList = [int(i) for i in input_string.split()]
        for i in userList:
            if i-1 not in list(range(len(subfolders))):
                print("no such dircetory, input invalid")
                return
        return [subfolders[i-1] for i in userList]


    def convert_matlab_to_csv(self, exp:str):
        """
        The expirement data is provided in the form of a MATLAB file, thus some pre-processing is needed
        in order to convert it to a more useable data-structre, in particular numpy array.
        Note that we convert the MATLAB file data to a pandas DataFrame and then we save it to a csv
        file for easier access in the future.
        @param exp: the experminet name
        @param pop: 0 for pursuit 1 for saccade
        @return:
        """
        path = self.__input_dir + exp + "/"
        folders = self.ask_for_dirs(path, MSG)
        for folder in folders:

            cell_names = fnmatch.filter(os.listdir(folder), '*.mat')  # filtering only the mat files.
            cell_names.sort()  # sorting the names of the files in order to create consistent runs.
            basename_folder = os.path.basename(os.path.dirname(folder))
            inner_folder = os.path.basename(folder)

            read_first_cell_in_kind = True

            for name in self.population_names:
                cells = self.filter_cells(cell_names, name)
                first_cell = True
                d=dict()
                y_axis_dict = dict()



                #need to create dictionary for labels to indices
                for cell in cells:
                    DATA_LOC = folder + "/" + cell  # cell file location
                    data = loadmat(DATA_LOC)  # loading the matlab data file to dict
                    spikes = data['data']['spikes'][0][0].todense().transpose()
                    print(spikes.shape)
                    y_axis_names = data['data'].dtype.names
                    for y_axis_name in y_axis_names:
                        try:
                            if y_axis_name == 'spikes':
                                continue
                            y_axis = data['data'][y_axis_name][0][0][0]
                            #check if the y_axis vector is has the same results as spikes
                            if (len(y_axis) != len(spikes)):
                                continue
                            unique_values = np.unique(y_axis)
                            if (first_cell):
                                d[y_axis_name] = dict()
                                for i, key in enumerate(unique_values):
                                    d[y_axis_name][key]=i
                            else:
                               if (len(unique_values) != len(d[y_axis_name].keys())):
                                   print("cell ", cell, " isnt valid because it has not all labels")
                                   return

                            y_axis = np.array([d[y_axis_name][label] for label in y_axis])
                            y_axis_dict[y_axis_name] = y_axis
                        except:
                            try :
                                del d[y_axis_name]
                            except:
                                continue
                            continue
                    first_cell = False



                    # saving the data to a csv file, and concatenating the number of samples from each file.
                    self.createDirectory("csv_files/" + basename_folder + "/" + inner_folder)
                    DataFrame(spikes).to_csv(self.__temp_path_for_writing + str(spikes.shape[0]).upper() + "#" + cell[:-3] + "csv")

                    #save the dict for the cell
                    with open(self.__temp_path_for_writing  + "." +  cell[:-4], 'wb') as info_file:
                        pickle.dump(y_axis_dict, info_file)

                    if (read_first_cell_in_kind):
                        with open(self.__temp_path_for_writing  + ".d", 'wb') as info_file:
                            pickle.dump(list(y_axis_dict.keys()), info_file)
                            read_first_cell_in_kind = False


    def savesInfo(self, info, pop_type, expirience_type):
        """
        Saves the information of the trials into file
        @param info: the results to be saved
        @param pop_type: the name of the population SNR MSN etc..
        @param expirience_type: eyes or reward
        @return:
        """
        with open(self.__temp_path_for_writing + pop_type + expirience_type, 'wb') as info_file:
            pickle.dump(info, info_file)

    def saveToLogger(self, name_of_file_to_write_to_logger):
        """
        save to logger the populations the alorithm finished
        @param name_of_file_to_write_to_logger:
        @param type:
        @return:
        """
        with open(self.__temp_path_for_writing + "Logger.txt", "a+") as info_file:
            info_file.write(name_of_file_to_write_to_logger + "\n")

    def loadFromLogger(self):
        """
        load from logger all the population the logger already finished with
        @param type:
        @return:
        """
        try:
            l = []
            with open(self.__temp_path_for_writing + "Logger.txt", "r") as info_file:
                for line in info_file.readlines():
                    l.append(line.rstrip().split('_')[0])
            return l
        except:
            return []

    def filterWithGaussian(self, X):
        """
        Smoothing the Matrix of trials
        @param X: the matrice needed to be smooth
        @return:
        """
        for i in range(len(X)):
            X[i] = gaussian_filter(X[i], sigma=self.SIGMA)
        return X

    def extractNSampelsFromOneDirection(self, direction):
        """
        pick randomly x number of trials to test from one direction  when x = self.number_of_cells_to_choose_for_test
        @param direction:
        @return:
        """
        np.random.shuffle(direction)
        test = direction[:self.number_of_cells_to_choose_for_test]
        train = direction[self.number_of_cells_to_choose_for_test:]
        return train, test

    def SortMatriceToListOfDirections(self, X, y):
        """
        Given a matrix of neural spikes and the direction w.r.t each spike,
        generates list of bundled spikes which corresponds to the same direction in each bundle.
        each index of the list corresponds to the direction of the eye movement.
        Also the number of spikes (vectors) in each index of the list (directions) = n,
        which is the minimum number of directions from all the other choosen cells.
        The way we choose cells is explained in the main function.
        @param X:
        @param y:
        @return:
        """
        directions = []
        for i in range(int(np.amax(y)+1)):
            idx = y == i
            temp = X[idx, :]
            directions.append(temp)
        return directions

    def extractNSampelsFromAllDirections(self, directions):
        """
        extract samples for test
        """
        directionsAverageVector = []
        testSampels = []
        for direction in directions:
            train, test = self.extractNSampelsFromOneDirection(direction)
            testSampels.append(test)
            averageVector = np.sum(np.array(train), axis=0) / train.shape[0]
            directionsAverageVector.append(averageVector)
        return np.vstack(directionsAverageVector), np.vstack(testSampels)

    def createTrainAndTestMatrice(self, X, y):
        """
        split the X matrice into 2 matrices. one for the train and one for the test
        @param X:
        @param y:
        @return:
        """
        directions = self.SortMatriceToListOfDirections(X, y)
        averageVectorsMatrice, testSampelsMatrice = self.extractNSampelsFromAllDirections(directions)
        return averageVectorsMatrice, testSampelsMatrice

     #if type is eyes so type =8
    def getTestVectors(self, type=8):
        """
        creates the test and train vectors. we already know them without the X train and test matrice therefore we
        made them automatically. if the experiment is 'eyes' we know that there is 8 direction vectors
        @param type: 8 or 2 depending on the experiment (8 driections or 2 rewards type)
        @return:
        """
        y_train = np.hstack([i for i in range(type)]).flatten()
        y_test = np.array(sum([[j for i in range(self.number_of_cells_to_choose_for_test)] for j in range(type)], []))
        return y_train, y_test

    def mergeSampeling1(self, loadFromDisk):
        """
        makes one matrice from all the cell names from loadFromDist list
        @param loadFromDisk: the
        @return:
        """
        TrainAvgMatricesCombined = []
        testMatriceCombined = []
        for X, y in loadFromDisk:
            averageVectorsMatrice, testSampelsMatrice = self.createTrainAndTestMatrice(X, y)
            TrainAvgMatricesCombined.append(averageVectorsMatrice)
            testMatriceCombined.append(testSampelsMatrice)
        return np.hstack(TrainAvgMatricesCombined), np.hstack(testMatriceCombined)

    def get_y_axis_from_disk(self, path, name, y_axis_key):
        """
            read the choses y_axis from disk or dictionary
        """
        try:
            return self.__files[name][y_axis_key]
        except:
            with open(path + name, 'rb') as info_file:
                info = pickle.load(info_file)
                self.__files[name] = info
                return info[y_axis_key]

    def clean_name(self,name):
        """
        return the name CRB_4863
        """
        name = name[name.find("#") + 1:]
        name = name[:name.find(".")]
        return name

    def read_from_disk_or_dictionary(self, path, cell_name):
        """
        return the cell spikes matrice from dictionary or from disk
        """
        try:
            return self.__files[cell_name]
        except:
            data =  pd.read_csv(path + cell_name)
            self.__files[cell_name] = data
            return data

    def read_from_disk(self, sampling, y_axis_key, is_fragments=False, segment=0, DIRECTION = True, ):
        """
        @param sampling: the names of the cells to read together and create one matrice
        @param is_fragments: to know if to split only the segmant or to read from 1000:2200
        @param segment:
        @param EYES: boolean - eyes or reward
        @return:
        """
        if (is_fragments):
            cut_first = self.LAG + (100 * segment)
            cut_last = self.LAG + (100 * (segment + 1))
        else:
            cut_first = self.LAG
            cut_last = LAST_COLUMN

        loadFiles = []
        for cell_name in sampling:
            dataset = self.read_from_disk_or_dictionary(self.temp_path_for_reading , cell_name)
            X = dataset.iloc[:, cut_first: cut_last].values
            y = self.get_y_axis_from_disk(self.temp_path_for_reading , "." + self.clean_name(cell_name), y_axis_key)
            if DIRECTION:
                X = self.filterWithGaussian(X)
            loadFiles.append((X, y))
        return loadFiles

    def filterCellsbyRows(self, cell_names):
        """
        filter the cells with lower bound of trials. if file is 148#SNR_4003 it means that this cell contain only 148
        trials
        @param cell_names:
        @return:
        """
        temp = []
        for cell_name in cell_names:
            new = cell_name[:cell_name.find("#")]
            if int(new) >= self.SAMPLES_LOWER_BOUND:
                temp.append(cell_name)
        return temp

    def control_group_cells(self, path):
        """
        inner function. check the simple knn algorithm validty.
        run only one cell each time and print the results
        path - absoult path os the folder containing the cells
        """
        self.temp_path_for_reading = path
        results = 0
        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(path), '*.csv')
        all_cell_names.sort()
        print(all_cell_names)
        classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')
        for cell in all_cell_names:
            # save the names of the cells and the score
            sum1 = 0
            # choose random K cells
            sampeling = [cell,]
            loadFiles = self.read_from_disk(sampeling, 'target_direction')
            for i in range(self.NUMBER_OF_ITERATIONS):
                X_train, X_test = self.mergeSampeling1(loadFiles)
                y_train, y_test = self.getTestVectors()

                classifier.fit(X_train, y_train)
                y_pred2 = classifier.predict(X_test)
                sum1 += accuracy_score(y_test, y_pred2)
            print(cell, sum1 / self.NUMBER_OF_ITERATIONS)
            results += sum1 / self.NUMBER_OF_ITERATIONS
        print(results / len(all_cell_names))


    def get_common_y_axis(self, folders, path):
        """
        function checks the .d file from all the folders and return only the common ones
        for example
        pursuit/.d -> ['reward', 'speed']
        saccade/.d -> ['reward', 'direction']
        will return only 'reward'
        """
        try:
            l = []
            for folder in folders:
                l += self.get_y_axis_values(folder + "/")
            return set(l)
        except:
            print("folder is currpted, delete folder of csv files and convert again")
            exit(1)

    def get_y_axis_column(self, common_y_axis):
        """
        return the axis the user chose
        """
        common_y_axis = list(common_y_axis)
        print("choose the dependent value\s:")
        for i, y in enumerate(common_y_axis):
            print(i + 1, ") ", y)
        input_string = input("enter the number of the depedent value\n")
        userList = [int(i) for i in input_string.split()]
        for i in userList:
            if i-1 not in list(range(len(common_y_axis))):
                print(i,"is not a valid index")
                return
        return [common_y_axis[i-1] for i in userList]



    def one_cell_session(self, all_cell_names, y_axis):
        """
        when k=1, instead of randomly chose TIMES cells it will run the algo over all the cells
        """
        results = 0
        classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')
        results_list = []
        for cell in all_cell_names:
            # save the names of the cells and the score
            sum1 = 0
            # choose random K cells
            sampeling = [cell, ]
            loadFiles = self.read_from_disk(sampeling, y_axis)
            for i in range(self.NUMBER_OF_ITERATIONS):
                X_train, X_test = self.mergeSampeling1(loadFiles)
                number_of_unique_labels = len(np.unique(loadFiles[0][1]))
                y_train, y_test = self.getTestVectors(number_of_unique_labels)
                classifier.fit(X_train, y_train)
                y_pred2 = classifier.predict(X_test)
                sum1 += accuracy_score(y_test, y_pred2)
            results_list.append((cell, sum1 / self.NUMBER_OF_ITERATIONS))
            results += sum1 / self.NUMBER_OF_ITERATIONS
        totalAv = results / len(all_cell_names)
        return results_list, totalAv

    def get_algos(self):
        """
        get use choice of algorithm
        """
        print("Choose the Algorithims")
        for i,algo in enumerate(self.__algo_names):
            print(i+1,") ", algo)
        input_string = input("enter all the algos numbers with space between them\n")
        userList = [int(i) for i in input_string.split()]
        for i in userList:
            if i-1 not in list(range(len(self.__algo_names))):
                print("no such dircetory, input invalid")
                return
        return [self.__algo_names[i-1] for i in userList]

    def analyze(self, project_name: str, lag: int, segments_size: int, is_common: bool = False):
        """
        @param project_name:  the name of the folder etc out/project_name
        @param lag: where to start the test in mili-seconds for exmaple, in direction expirement
                lag=1000(start of the expirement)
        @param segments_size: how many segments to cut (each segment is 100 ms), relevant for simple_knn_fragments only
        @param is_common:
        @return:
        """
        self.LAG = lag
        self.SEGMENTS = segments_size

        path = self.__output_dir + "csv_files/" + project_name + "/"
        folders = self.ask_for_dirs(path, RUNNING_THE_TEST)
        common_y_axis = self.get_common_y_axis(folders, path)

        y_axis_keys = self.get_y_axis_column(common_y_axis)

        algos = self.get_algos()
        for algo in algos:
            self.ALGOS[algo](y_axis_keys, folders, is_common=is_common)

    def filter_cells_for_common(self, path, folder_name, folders, is_common):
        if (is_common and len(folders) > 1):
            others = []
            current =  fnmatch.filter(os.listdir(folder_name), '*.csv')
            for folder in folders:
                if folder == folder_name:
                    pass
                else:
                    others += fnmatch.filter(os.listdir(folder), '*.csv')
            others = [decoder.get_population_name_and_population(name) for name in others]
            return [name for name in current if decoder.get_population_name_and_population(name) in others]
        else:
            return fnmatch.filter(os.listdir(path), '*.csv')

    @staticmethod
    def create_name_of_folder(folders):
        """
        get list of folder names for example ['pursuit','saccade'] and return
        'pursuit_saccade'
        """
        name = ["common"]
        for folder in folders:
            name.append(os.path.basename(folder))
        return "_".join(name)

    def simple_knn(self, y_axis_keys, folders, is_common):
        for y_axis_key in y_axis_keys:
            for folder in folders:
                basename_folder = os.path.basename(os.path.dirname(folder))

                self.temp_path_for_reading = folder + "/"

                common_path  = ""
                if (is_common and len(folders)>1):
                    common_path = decoder.create_name_of_folder(folders) + "/"
                inner_folder = os.path.basename(folder)

                self.createDirectory( basename_folder +  "/" + y_axis_key + "/" + common_path + inner_folder + "/simple_knn/")
                # loading folder
                all_cell_names = self.filter_cells_for_common(self.temp_path_for_reading, folder, folders, is_common)
                all_cell_names.sort()
                #empty files cach
                self.__files = dict()
                self.find_already_made_files()
                for population in tqdm([x for x in self.population_names if x not in self.loadFromLogger()],
                                       desc="Processing folder " + folder):
                    cell_names = self.filter_cells(all_cell_names, population)
                    cell_names = self.filterCellsbyRows(cell_names)
                    # build list which saves info
                    info = []

                    if (self.K > len(cell_names) - 1):
                        self.K = len(cell_names) - 1

                    # saves the rate of the success for each k population
                    sums = []
                    classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')
                    # iterating over k-population of cells from 1 to K
                    for number_of_cells in tqdm(range(1, self.K + 1, self.step),
                                                desc="Processing population " + population):
                        if number_of_cells == 1:
                            infoPerGroupOfCells, totalAv = self.one_cell_session(cell_names, y_axis_key)
                            info.append((infoPerGroupOfCells, totalAv))
                        else:
                            # saves each groupCells
                            infoPerGroupOfCells = []

                            # intializing counter
                            totalAv = 0

                            # iterating TImes for solid average
                            for j in range(self.TIMES):
                                # save the names of the cells and the score
                                scoreForCells = []

                                sum1 = 0
                                # choose random K cells
                                sampeling = random.sample(cell_names, k=number_of_cells)
                                loadFiles = self.read_from_disk(sampeling, y_axis_key)
                                for i in range(self.NUMBER_OF_ITERATIONS):
                                    X_train, X_test = self.mergeSampeling1(loadFiles)
                                    number_of_unique_labels = len(np.unique(loadFiles[0][1]))
                                    y_train, y_test = self.getTestVectors(number_of_unique_labels)

                                    classifier.fit(X_train, y_train)

                                    # check algo validty
                                    # np.random.shuffle(y_test)

                                    y_pred2 = classifier.predict(X_test)
                                    sum1 += accuracy_score(y_test, y_pred2)

                                totalAv += sum1 / self.NUMBER_OF_ITERATIONS
                                scoreForCells.append((sampeling, sum1 / self.NUMBER_OF_ITERATIONS))
                                infoPerGroupOfCells.append(scoreForCells)

                            info.append((infoPerGroupOfCells, totalAv / self.TIMES))

                    self.savesInfo(info, population, "")
                    self.saveToLogger(population)
                decoder.save_parametes_in_text(self)

    @staticmethod
    def save_parametes_in_text(a):
        d = {"_decoder__input_dir" : "Input Dir", "_decoder__output_dir": "Output Dir", "population_names": "Population Names"
            , "_decoder__temp_path_for_writing" : "", "_decoder__files": "", "ALGOS":"",
             "LAG":"Begining of The Expirement(in ms)", "SEGMENTS":"The Numbers of Segments", "temp_path_for_reading":""}
        with open(a.__temp_path_for_writing + "param.txt", "w") as param_file:
            for key,value in vars(a).items():
                title = d[key]
                if (title != ""):
                    param_file.write(title +": " + str(value) + "\n")



    def createDirectory(self, name):
        if not os.path.exists(self.__output_dir + name):
            os.makedirs(self.__output_dir + name)
        self.__temp_path_for_writing = self.__output_dir + name + "/"

    def find_already_made_files(self):
        for x in self.population_names:
            if x in self.loadFromLogger():
                print(x, "is already done!")


    def simple_knn_fragments(self, y_axis_keys, folders, is_common):
        for y_axis_key in y_axis_keys:
            for folder in folders:
                basename_folder = os.path.basename(os.path.dirname(folder))

                common_path  = ""
                if (is_common and len(folders)>1):
                    common_path = decoder.create_name_of_folder(folders) + "/"

                inner_folder = os.path.basename(folder)
                self.temp_path_for_reading = folder + "/"
                self.createDirectory(basename_folder + "/" + y_axis_key + "/"  +common_path +
                                     inner_folder + "/simple_knn_fragments/")
                # loading folder
                # all_cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
                all_cell_names = self.filter_cells_for_common(self.temp_path_for_reading, folder, folders, is_common)

                all_cell_names.sort()

                # empty files cach
                self.__files = dict()
                self.find_already_made_files()
                for population in tqdm([x for x in self.population_names if x not in self.loadFromLogger()],
                                       desc="Processing folder " + os.path.basename(folder)):
                    cell_names = self.filter_cells(all_cell_names, population)
                    cell_names = self.filterCellsbyRows(cell_names)
                    # build list which saves info
                    info = []
                    if (self.K > len(cell_names) - 1):
                        self.K = len(cell_names) - 1

                    # saves the rate of the success for each k population
                    sums = []
                    classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2,
                                                      weights='distance')
                    # iterating over k-population of cells from 1 to K
                    for i in tqdm(range(self.SEGMENTS), desc="Processing " + population + " segments"):
                        sums = []
                        info = []
                        segment = i
                        for number_of_cells in range(1, self.K + 1, self.step):
                            # saves each groupCells
                            infoPerGroupOfCells = []

                            # intializing counter
                            totalAv = 0
                            for j in range(self.TIMES):
                                # save the names of the cells and the score
                                scoreForCells = []
                                sum = 0
                                # choose random K cells
                                sampeling = random.sample(cell_names, k=number_of_cells)
                                loadFiles = self.read_from_disk(sampeling, y_axis_key, is_fragments=True, segment=segment)

                                for i in range(self.NUMBER_OF_ITERATIONS):
                                    X_train, X_test = self.mergeSampeling1(loadFiles)
                                    number_of_unique_labels = len(np.unique(loadFiles[0][1]))
                                    y_train, y_test = self.getTestVectors(number_of_unique_labels)
                                    classifier.fit(X_train, y_train)
                                    y_pred2 = classifier.predict(X_test)
                                    # np.random.shuffle(y_test)
                                    sum += accuracy_score(y_test, y_pred2)
                                totalAv += sum / self.NUMBER_OF_ITERATIONS
                                scoreForCells.append((sampeling, sum / self.NUMBER_OF_ITERATIONS))
                                infoPerGroupOfCells.append(scoreForCells)
                            info.append((infoPerGroupOfCells, totalAv / self.TIMES))
                            sums.append(totalAv / self.TIMES)
                            self.savesInfo(info, population, str(segment))
                    self.saveToLogger(population)
                decoder.save_parametes_in_text(self)


    @staticmethod
    def file_name_changer(path):
        """
        helper func for name changing
        @param path:
        @return:
        """
        path = os.path.join(path, '')
        # reduce PC and BG in the begining
        for reg in ['*PC*','*BG*']:
            all_cell_names = fnmatch.filter(os.listdir(path), reg)
            for name in all_cell_names:
                newName = name[3:]
                os.rename(path + name, path + newName)

        # makes file name captial
        all_cell_names = fnmatch.filter(os.listdir(path), '*.mat')
        for name in all_cell_names:
            l = name.split('.')
            newName = l[0].upper() + "." + l[1]
            os.rename(path + name, path + newName)

    def help(self):
        with open("essentials/decoder_instructions", 'r') as info_file:
            for line in info_file.readlines():
                print(line)

    def get_y_axis_values(self,path : str):
        """
        this function open the info file stored ad .d with all the y_axises from the matlab folder
        """
        with open(path + ".d", 'rb') as info_file:
            info = pickle.load(info_file)
        return info

    @staticmethod
    def get_name_from_path(path:str):
        """
          take the name from the path
         from ~/Neural_Analyzer/out/nogas_project/target_direction/pursuit/simple_knn/SNR
         return SNR
        """
        return os.path.basename(path)

    @staticmethod
    def get_kind_from_path(path:str):
        """
          take the name from the path
         from ~/Neural_Analyzer/out/nogas_project/target_direction/pursuit/simple_knn/SNR
         return pursuit
        """
        return  os.path.basename(os.path.dirname(os.path.dirname(path)))

    @staticmethod
    def get_expirement_from_path(path:str):
        """
          take the name from the path
         from ~/Neural_Analyzer/out/nogas_project/target_direction/pursuit/simple_knn/SNR
         return target_direction
        """
        return  os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))

    @staticmethod
    def get_algo_name_from_path(path):
        """
          take the name from the path
         from ~/Neural_Analyzer/out/nogas_project/target_direction/pursuit/simple_knn/SNR
         return simple_knn
        """
        return os.path.basename(os.path.dirname(path))

    @staticmethod
    def get_full_name(path):
        return decoder.get_expirement_from_path(path) + " " +\
                decoder.get_kind_from_path(path) + " " + \
                decoder.get_algo_name_from_path(path) + " " + \
                decoder.get_name_from_path(path)


    #better to use DataFrame Builtin function to_csv and read it as table in matlab with
    #  in matlab write : A = readtable("filename.csv")
    @staticmethod
    def save_df_to_mat(data,out_path, the_name_you_want):
        out_path = os.path.join(out_path, '')
        df = data.apply(tuple).to_dict()
        mp.savemat(out_path + the_name_you_want, {'structs': df})

