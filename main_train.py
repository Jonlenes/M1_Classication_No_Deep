import pandas
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from post_processing import generate_submit_simple
from dataset import CameraDataset
from pre_process_dataset import path_load_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from post_processing import CLASSES
import matplotlib.pyplot as plt

path_dataset = "/home/jonlenes/Desktop/dataset_csv/"


def load_csv_file(file_name):
    """
    Carrega csv localizado em path_dataset
    """
    return pandas.read_csv(path_dataset + file_name)


def load_dataset_train():
    """
    Carrega os dados de treinamento
    """
    df_train = load_csv_file("31_train.csv")
    df_train_target = load_csv_file("31_target_train.csv")

    return df_train.values, df_train_target.values


def load_dataset_test():
    """
    Carrega csv de test
    """
    df_test = load_csv_file("31_test.csv")
    return df_test.values


def only_unalt(x):
    """
    Retorna somente as imagens do dataset que não foram alteradas
    """
    files = CameraDataset(path_load_dataset + "test_files", path_load_dataset, None).get_files_name()
    unalt_files = []
    for i in range(len(files)):
        # if "_manip" in files[i]:
        if "_unalt" in files[i]:
            unalt_files.append(i)

    x = list(map(lambda unalt_index: x[unalt_index], unalt_files))
    files = list(map(lambda unalt_index: files[unalt_index], unalt_files))

    return files, x


def get_best_coef(coef):
    """
    Dados os coeficientes, retorna os melhores
    """
    best_coef_index = []
    for c in coef:
        l = list(enumerate(c))
        l.sort(key=lambda v: v[1])
        best_coef_index.append(l[len(l) - 1][0])
    return best_coef_index


def exp_x(x, best_coef_index):
    """
    Elava um determinado feature a uma potencia 
    """
    for idx in best_coef_index:
        x = np.append(x, (x[:, idx] ** 10).reshape(x.shape[0], 1), axis=1)
        print(x.shape)
    return x


def load_model_complexity():
    """
    Pega o features com melhor coeficiente e o adiciona novamente 
    com maior complexidade (Potência)
    """
    x, y = load_dataset_train()

    x = scale(x)
    clf = LogisticRegression(intercept_scaling=1.5)
    clf.fit(x, y.ravel())

    best_coef_index = get_best_coef(clf.coef_)
    x = exp_x(x, best_coef_index)

    x_test = load_dataset_test()
    x_test = exp_x(x_test, best_coef_index)
    return mean_normalisation(x), y, mean_normalisation(x_test)


def mean_normalisation(x):
    """
    Normalização
    """
    for j in range(x.shape[1]):
        arr = x[:, j]
        min_val = np.min(arr)
        max_val = np.max(arr)
        mean = np.mean(arr)
        x[:, j] = (x[:, j] - mean) / (max_val - min_val)
    return x


def MLPClassifier_test_num_layers():
    print("Starting...")
    load_model_exp = False

    if load_model_exp:
        x_train, y_train, x_test = load_model_complexity()
    else:
        x_train, y_train = load_dataset_train()
        x_test = load_dataset_test()

    print("Features:", x_train.shape)

    x_train = scale(x_train)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    x_test = scale(x_test)

    layers = []
    accs = []

    for i in range(25, 1001, 25):
        clf = MLPClassifier(hidden_layer_sizes=(i, ), max_iter=10000)
        clf.fit(x_train, y_train.ravel())

        y_pred = clf.predict(x_validation)
        score = accuracy_score(y_validation, y_pred)
        print(i, score * 100)

        layers.append(i)
        accs.append(score * 100)

    # plt = PlotFunction()
    # plt.add_function(layers, accs)

    np.savetxt("layers.txt", layers, fmt='%.0f')
    np.savetxt("accs.txt", accs, fmt='%.2f')

def main():
    print("Starting...")
    load_model_exp = False

    if load_model_exp:
        x_train, y_train, x_test = load_model_complexity()
    else:
        x_train, y_train = load_dataset_train()
        x_test = load_dataset_test()

    print("Features:", x_train.shape)

    x_train = scale(x_train)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    # Escolhendo o classificador
    # clf = LogisticRegression()
    # clf = KNeighborsClassifier()
    clf = MLPClassifier(hidden_layer_sizes=(275, ), max_iter=10000)
    clf.fit(x_train, y_train.ravel())

    # Testando o modelo
    y_pred = clf.predict(x_validation)
    print(accuracy_score(y_validation, y_pred))
    print(classification_report(y_validation, y_pred, target_names=CLASSES))

    # dados de teste
    x_test = scale(x_test)
    test_pred = clf.predict(x_test)

    # Gerando csv para submeter no Kaggle
    files = CameraDataset(path_load_dataset + "test_files", path_load_dataset, None).get_files_name()
    generate_submit_simple(test_pred, files, path_dataset + "0_submit_200.csv")


if __name__ == '__main__':
    main()


