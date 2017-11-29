# ITERATIVE SOFT THRESHOLDING or ISTA
import numpy as np
import copy
import math

# L1 minimization
# l = number of users, r = number of items, k = number of intermediate factors
num_users = 943
# num_items = 4
num_items = 1682
k = 30
lambda_reg = 0.01
count = 21
Y = np.zeros((num_users, num_items))
B = np.zeros((num_users, num_items))
print "k: ", k
print "lambda: ", lambda_reg
print "epochs: ", count
# Y is the original rating matrix, B is the binary matrix
# Y_processed matrix will contain the mean in place of missing values

with open('/home/nehaj/Downloads/Datasets/movielens-100k/u1.base') as training_file:
    for line in training_file:
        line = line.split('\t')
        x = int(line[0])-1
        y = int(line[1])-1
        Y[x][y] = int(line[2])
        # print(ui_matrix[x][y])
print(Y)


def initialize(Y, B):
    global num_items, num_users
    processed = np.zeros((num_users, num_items))
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            if Y[i][j] == 0:
                row_mean = np.mean(Y[i, :])
                col_mean = np.mean(Y[:, j])
                ui_mean = (row_mean + col_mean)/2.0
                processed[i][j] = ui_mean
            else:
                B[i][j] = 1
                processed[i][j] = Y[i][j]
    return processed, B

def alternatingMinimization(matrix, lambda_reg):
    global count, k
    global num_users, num_items
    LAMB = np.zeros((k, k))
    np.fill_diagonal(LAMB, lambda_reg)
    # print(LAMB)
    U = np.random.rand(len(matrix), k)
    for c in range(count):
        # print("c in altMin: ", c)
        term1 = np.linalg.lstsq(np.dot(np.transpose(U), U), np.dot(np.transpose(U), matrix))[0]
        # term2 = np.dot(l1_minimization(matrix, U, lambda_reg), LAMB)
        term2 = l1_minimization(matrix, U, lambda_reg)
        # V = term1 + np.transpose(term2)
        V = np.transpose(term2)
        V[V < 0] = 0
        U_trans = np.linalg.lstsq(np.dot(V, np.transpose(V)) + LAMB, np.dot(V, np.transpose(matrix)))[0]
        U_trans[U_trans < 0] = 0
        U = np.transpose(U_trans)
        if np.array_equal(np.dot(U, V), matrix):
            break

    final = np.dot(U, V)
    return final

def lfm(Y, Y_processed, B, lambda_reg):
    global count
    c = count
    while c > 0:
        print("lfm count:", c)
        intermediate_matrix = Y_processed + (Y - np.multiply(B, Y_processed))
        Y_processed = alternatingMinimization(intermediate_matrix, lambda_reg)
        c -= 1

    return Y_processed

def l1_minimization(matrix, U, lambda_reg):
    global count
    global k, num_users, num_items
    V = np.zeros((num_items, k))
    product = np.dot(U, np.transpose(U))
    eig_values, eig_vector = np.linalg.eig(product)
    alpha = np.amax(eig_values)
    for c in range(count):
        # print("c in l1_min: ", c)
        UT = np.transpose(U)
        VT = np.transpose(V)
        tmp = (matrix - np.dot(U, VT))
        T = V + (1/alpha) * np.transpose(np.dot(UT, tmp))
        # print(T)
        for i in range(len(T)):
            for j in range(len(T[0])):
                value = math.fabs(T[i][j]) - (lambda_reg / (2.0 * alpha))
                # print(value)
                if value < 0.0:
                    value = 0.0
                if T[i][j] >= 0:
                    V[i][j] = value
                else:
                    V[i][j] = (-1) * value
    return V


Y_processed, B = initialize(Y, B)

print("Y_processed-----------------")
print(Y_processed)
print(B)

final_rating = lfm(Y, Y_processed, B, lambda_reg)
print("Y:------------------------")
print(Y)
print "Final matrix: "
print final_rating

num_test = 0
mean_error = 0
def printNMAE(val):
    nmae = val / 4.0
    print "nmae is:"
    print(nmae)


with open('/home/nehaj/Downloads/Datasets/movielens-100k/u1.test') as testing_file:
    for line in testing_file:
        line = line.split('\t')
        x_test = int(line[0]) - 1
        y_test = int(line[1]) - 1
        num_test += 1
        mean_error += math.fabs(final_rating[x_test][y_test] - int(line[2]))

    intermediate = (mean_error / num_test)
    print("mae is:", intermediate)
    printNMAE(intermediate)

print "k: ", k
print "lambda: ", lambda_reg
print "epochs: ", count




