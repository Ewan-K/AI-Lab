# 导入必要的包
import matplotlib.pyplot as plt
import numpy as np
# import cv2
# from PIL import Image
# import os


def show_img(img, h=112, w=92):
    """
    展示单张图片

    :param img: numpy array 格式的图片
    :return:
    """
    # 展示图片
    plt.imshow(img.reshape(h, w), 'gray')
    plt.axis('off')
    plt.show()


def plot_gallery(images, titles, n_row=3, n_col=5, h=112, w=92):  # 3行4列
    """
    展示多张图片

    :param images: numpy array 格式的图片
    :param titles: 图片标题
    :param h: 图像reshape的高
    :param w: 图像reshape的宽
    :param n_row: 展示行数
    :param n_col: 展示列数
    :return:
    """
    # 展示图片
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def spilt_data(nPerson, nPicture, data, label):
    """
    分割数据集

    :param nPerson : 志愿者数量
    :param nPicture: 各志愿者选入训练集的照片数量
    :param data : 等待分割的数据集
    :param label: 对应数据集的标签
    :return: 训练集, 训练集标签, 测试集, 测试集标签
    """
    # 数据集大小和意义
    allPerson, allPicture, rows, cols = data.shape

    # 划分训练集和测试集
    train = data[:nPerson, :nPicture, :, :].reshape(nPerson * nPicture,
                                                    rows * cols)
    train_label = label[:nPerson, :nPicture].reshape(nPerson * nPicture)
    test = data[:nPerson,
                nPicture:, :, :].reshape(nPerson * (allPicture - nPicture),
                                         rows * cols)
    test_label = label[:nPerson,
                       nPicture:].reshape(nPerson * (allPicture - nPicture))

    # 返回: 训练集, 训练集标签, 测试集, 测试集标签
    return train, train_label, test, test_label


def plot_gallery(images, titles, n_row=3, n_col=5, h=112, w=92):  # 3行4列
    """
    展示多张图片

    :param images: numpy array 格式的图片
    :param titles: 图片标题
    :param h: 图像reshape的高
    :param w: 图像reshape的宽
    :param n_row: 展示行数
    :param n_col: 展示列数
    :return:
    """
    # 展示图片
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def eigen_train(trainset, k=20):
    """
    训练特征脸（eigenface）算法的实现

    :param trainset: 使用 get_images 函数得到的处理好的人脸数据训练集
    :param K: 希望提取的主特征数
    :return: 训练数据的平均脸, 特征脸向量, 中心化训练数据
    """

    ###############################################################################
    ####                   训练特征脸（eigenface）算法的实现                     ####
    ####                        请勿修改该函数的输入输出                         ####
    ###############################################################################
    #                                                                             #

    #  compute mean face
    avg_img = np.zeros((1, 112 * 92))

    for i in trainset:
        avg_img = np.add(avg_img, i)

    avg_img = np.divide(avg_img, float(k)).flatten()
    # plt.imshow(avg_img.reshape(112, 92), cmap='gray')
    # plt.tick_params(labelleft='off',
    #                 labelbottom='off',
    #                 bottom='off',
    #                 top='off',
    #                 right='off',
    #                 left='off',
    #                 which='both')
    # plt.show()

    #  compute normalised face of trainset
    normalised_training_tensor = np.ndarray(shape=(k, 112 * 92))
    for i in range(k):
        normalised_training_tensor[i] = np.subtract(trainset[i], avg_img)

    # compute covariance matrix
    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix, 8.0)
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    # print('Eigenvectors of Cov(X): \n%s' % eigenvectors)
    # print('\nEigenvalues of Cov(X): \n%s' % eigenvalues)
    eig_pairs = [(eigenvalues[index], eigenvectors[:, index])
                 for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [
        eig_pairs[index][1] for index in range(len(eigenvalues))
    ]
    reduced_data = np.array(eigvectors_sort[:k]).transpose()
    # print(trainset.shape, trainset.transpose().shape)
    # print(reduced_data.shape)
    trainset = trainset[:k]  # In case the trainset passed in is not as desired, truncate to k rows
    feature = np.dot(trainset.transpose(), reduced_data)
    feature = feature.transpose()

    norm_img = normalised_training_tensor
    #                                                                             #
    ###############################################################################
    #############           在生成 main 文件时, 请勾选该模块            #############
    ###############################################################################

    # 返回：平均人脸、特征人脸、中心化人脸
    return avg_img, feature, norm_img


def rep_face(image, avg_img, eigenface_vects, numComponents=0):
    """
    用特征脸（eigenface）算法对输入数据进行投影映射，得到使用特征脸向量表示的数据

    :param image: 输入数据
    :param avg_img: 训练集的平均人脸数据
    :param eigenface_vects: 特征脸向量
    :param numComponents: 选用的特征脸数量
    :return: 输入数据的特征向量表示, 最终使用的特征脸数量
    """

    ###################################################################################
    ####  用特征脸（eigenface）算法对输入数据进行投影映射，得到使用特征脸向量表示的数据  ####
    ####                          请勿修改该函数的输入输出                           ####
    ###################################################################################
    #                                                                                 #
    eigenface_vects_reduced = eigenface_vects[:numComponents]
    representation = np.dot(eigenface_vects_reduced, image)
    numEigenFaces = numComponents
    #                                                                                 #
    ###################################################################################
    #############             在生成 main 文件时, 请勾选该模块              #############
    ###################################################################################

    # 返回：输入数据的特征向量表示, 特征脸使用数量
    return representation, numEigenFaces


def recFace(representations,
            avg_img,
            eigenVectors,
            numComponents,
            sz=(112, 92)):
    """
    利用特征人脸重建原始人脸

    :param representations: 表征数据
    :param avg_img: 训练集的平均人脸数据
    :param eigenface_vects: 特征脸向量
    :param numComponents: 选用的特征脸数量
    :param sz: 原始图片大小
    :return: 重建人脸, str 使用的特征人脸数量
    """

    ###############################################################################
    ####                        利用特征人脸重建原始人脸                         ####
    ####                        请勿修改该函数的输入输出                         ####
    ###############################################################################
    #                                                                             #
    face = avg_img + np.dot(representations, eigenVectors[:numComponents])
    # for i in range(0, numComponents):
    #     print(representations.shape, eigenVectors[i].shape)
    #     face = np.add(face, np.dot(representations, eigenVectors[i]))

    #                                                                             #
    ###############################################################################
    #############           在生成 main 文件时, 请勾选该模块            #############
    ###############################################################################

    # 返回: 重建人脸, str 使用的特征人脸数量
    return face, 'numEigenFaces_{}'.format(numComponents)


datapath = './ORL.npz'
ORL = np.load(datapath)
data = ORL['data']
label = ORL['label']
num_eigenface = 200

train_vectors, train_labels, test_vectors, test_labels = spilt_data(40, 5, data, label)
train_vectors = train_vectors / 255
print(train_vectors.shape)
test_vectors = test_vectors / 255

# print("训练数据集:", train_vectors.shape)
# print("测试数据集:", test_vectors.shape)

# # 展示单张图片
# show_img(train_vectors[0])

# # 展示多张图片
# plot_gallery(train_vectors, train_labels)

# 返回平均人脸、特征人脸、中心化人脸
avg_img, eigenface_vects, trainset_vects = eigen_train(train_vectors, num_eigenface)

# # 打印两张特征人脸作为展示
# eigenfaces = eigenface_vects.reshape((num_eigenface, 112, 92))
# eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# plot_gallery(eigenfaces, eigenface_titles, n_row=1, n_col=2)

# train_reps = []
# for img in train_vectors:
#     train_rep, _ = rep_face(img, avg_img, eigenface_vects, num_eigenface)
#     train_reps.append(train_rep)

# num = 0
# for idx, image in enumerate(test_vectors):
#     label = test_labels[idx]
#     test_rep, _ = rep_face(image, avg_img, eigenface_vects, num_eigenface)

#     results = []
#     for train_rep in train_reps:
#         similarity = np.sum(np.square(train_rep - test_rep))
#         results.append(similarity)
#     results = np.array(results)

#     if label == np.argmin(results) // 5 + 1:
#         num = num + 1

# print("人脸识别准确率: {}%".format(num / 80 * 100))

print("重建训练集人脸")
# 读取train数据
image = train_vectors[100]

faces = []
names = []
# 选用不同数量的特征人脸重建人脸
for i in range(20, 200, 20):
    representations, numEigenFaces = rep_face(image, avg_img, eigenface_vects, i)
    face, name = recFace(representations, avg_img, eigenface_vects, numEigenFaces)
    faces.append(face)
    names.append(name)

plot_gallery(faces, names, n_row=3, n_col=3)

print("-" * 55)
print("重建测试集人脸")
# 读取test数据
image = test_vectors[54]

faces = []
names = []
# 选用不同数量的特征人脸重建人脸
for i in range(20, 200, 20):
    representations, numEigenFaces = rep_face(image, avg_img, eigenface_vects, i)
    face, name = recFace(representations, avg_img, eigenface_vects, numEigenFaces)
    faces.append(face)
    names.append(name)

plot_gallery(faces, names, n_row=3, n_col=3)
