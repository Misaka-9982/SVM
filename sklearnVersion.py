from sklearn import svm, metrics
from main import load_data

t_data, t_label, v_data, v_label = load_data()
model = svm.SVC(C=0.8, kernel='rbf')
model.fit(t_data, t_label)
pre = model.predict(v_data)
print(metrics.accuracy_score(v_label, pre))
print(model._gamma)
