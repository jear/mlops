import matplotlib.pyplot as plt
import numpy as np

def get_insight_of_data(labels, nbLabels = 22) :
	"""
	Compute the number of labels of each classes

	"""
	statistic = np.zeros(nbLabels)
	for j in range(0, len(labels)) :
		for k in range(0,len(labels[j])) :
			if labels[j][k] != 0 :
				statistic[k] += 1
	return statistic

def plot_data(stats, nbLabels = 22) :
	"""
	Plot the number of labels of each classes

	"""
	b = [i for i in range(0,nbLabels)]
	width = 0.3
	plt.bar(b, stats, width, color ="b")  # arguments are passed to np.histogram
	plt.title("Stats of the dataset labels")
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 15
	fig_size[1] = 15
	plt.show()




def plot_model_training_metrics(history, metric = 'acc', save = False , folderWithName = 'C:/Documents/modelTest.png' ) :
	"""
    Plot the metric over the epoch of training

    history : model keras,
        Trained model over epochs

    metric : str, optional
        Metric to plot.
        Default value, "acc".

    save : boolean, optional.
		Weither to plot or not
        Default value : False

    folderWithName : str, optional
        Location with name where you want to save.
		If save is False it will have no effect
        Default value, 'categorical_crossentropy'.


    """
	plt.plot(history.history[metric])
	plt.plot(history.history['val_' + metric])
	plt.title('Model ' + metric)
	plt.ylabel(metric)
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper left')
	if save :
		plt.savefig(folderWithName)
	#plt.show()