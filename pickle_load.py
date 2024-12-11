import pickle

with open('/mnt/c/Users/PVRL-01/Documents/Donald Intal/mlproj/plotoutputSVM/best_svm_params.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)